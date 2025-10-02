#!/usr/bin/env python3
"""
chat_engine.py – calendar-aware + RAG-aware chat engine using OCI ADK.
Auto-routes non-calendar intents to RAG by default.
Keeps calendar flows intact.
"""

from __future__ import annotations

import os
import re
import argparse
import logging
import asyncio
from typing import Generator, Optional

from oci.addons.adk import Agent, AgentClient

from calendar_toolkit import CalendarToolkit
from rag_toolkit import RagToolkit
from load_config import LoadConfig


# What counts as calendar intent
_CALENDAR_RE = re.compile(
    r"\b("
    r"calendar|schedule|meeting|meetings|event|events|invite|attendees?|"
    r"busy|free|availability|when|what time|where|room|today|tomorrow|"
    r"this week|next week|next \d+\s*(days?|hours?)"
    r")\b",
    re.IGNORECASE,
)

# Short follow ups that stay on calendar
_FOLLOWUP_RE = re.compile(
    r"\b(what time|when exactly|how long|how far|how many minutes|where is it|"
    r"where|with who|attendees?|link|join|location)\b",
    re.IGNORECASE,
)

# Optional user route tag
_ROUTE_TAG_RE = re.compile(r"^\s*\[ROUTE:(RAG|CAL)\]\s*", re.IGNORECASE)

# Token capture, three forms
_MS_TOKEN_PATTERNS = [
    # Explicit bracket form: [MS_TOKEN: token]
    re.compile(r"\[\s*MS_TOKEN\s*:\s*(?P<token>[^ \]\r\n]+)\s*\]", re.IGNORECASE),
    # Inline assignment: MS_TOKEN=token
    re.compile(r"\bMS_TOKEN\s*=\s*(?P<token>\S+)", re.IGNORECASE),
    # Chat command: set ms token <token>
    re.compile(r"^\s*set\s+ms\s+token\s+(?P<token>\S+)\s*$", re.IGNORECASE),
    # Bearer header style: Bearer <jwt>
    re.compile(r"\bBearer\s+(?P<token>[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)", re.IGNORECASE),
    # Raw JWT pasted anywhere in the message (common for MS Graph tokens).
    re.compile(r"(?P<token>[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"),
]

TOKEN_PROMPT_HTML = (
    "I need your Microsoft Graph access token to read your calendar. "
    "You can paste the raw token string directly (or use `[MS_TOKEN: YOUR_TOKEN]`)."
    "<br><br>"
    "To get a Microsoft Graph access token:"
    "<br>- Go to https://developer.microsoft.com/en-us/graph/graph-explorer"
    "<br>- Sign in (top right corner)"
    "<br>- Click the Access token tab"
    "<br>- Copy the token"
)

# Shared toolkits
_calendar_toolkit = CalendarToolkit()
_rag_toolkit = RagToolkit()


def build_agent() -> Agent:
    properties = LoadConfig()
    profile_name = properties.getDefaultProfile()
    agent_endpoint_ocid = properties.getAgentEndpointOcid()
    oci_region = properties.getAgentRegion()

    if not agent_endpoint_ocid or not agent_endpoint_ocid.startswith("ocid1.genaiagentendpoint."):
        raise ValueError("LoadConfig().getAgentEndpointOcid() must return a valid agent-endpoint OCID.")

    client = AgentClient(auth_type="api_key", profile=profile_name, region=oci_region)

    instructions = (
        """<role>
You are a multi-tool assistant with two capabilities:
1) Calendar assistant (timeframe-gated Microsoft 365 reads).
2) RAG assistant (answer general questions using the rag toolkit).
</role>

<routing-protocol>
- If the user message begins with "[ROUTE:RAG]" use the RAG toolkit only.
- If the user message begins with "[ROUTE:CAL]" use the calendar flow only.
- Otherwise, if the query is clearly about calendar or scheduling, use the calendar flow. For anything else, prefer the RAG toolkit.
</routing-protocol>

<calendar-flow>
- Never fetch calendar without a valid time window, max 7 days. If missing, ask a concise follow-up to get a window.
- Call resolve_timeframe(timeframe), then fetch_calendar_events(start_datetime, end_datetime).
- Do not ask for timezone. The toolkit normalises times and includes the zone name in results.
- Short follow-ups like “with who?”, “where is it?”, “what time exactly?”, and “how long from now?” refer to the last calendar results.
</calendar-flow>

<rag-flow>
- For non-calendar questions, call the RAG toolkit once with the user question and return that output.
</rag-flow>

<format>
Return clean, simple HTML (no CSS/JS). Use concise paragraphs and lists.
</format>"""
    )

    return Agent(
        client=client,
        agent_endpoint_id=agent_endpoint_ocid,
        instructions=instructions,
        tools=[_rag_toolkit, _calendar_toolkit],
    )


calendar_agent: Agent = build_agent()


class ChatEngine:
    def __init__(self, debug: bool = False):
        self._agent = calendar_agent
        self.debug = debug
        self._session_id: Optional[str] = None
        self._last_domain: Optional[str] = None  # 'calendar' | 'rag'
        self._pending_calendar_msg: Optional[str] = None  # last calendar ask waiting for token
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    def _infer_domain(self, text: str) -> str:
        t = (text or "").strip()
        if len(t) <= 24 and self._last_domain:
            return self._last_domain
        if _CALENDAR_RE.search(t):
            return "calendar"
        return "rag"

    def _prefix_for_domain(self, domain: str) -> str:
        return "[ROUTE:CAL] " if domain == "calendar" else "[ROUTE:RAG] "

    def _maybe_capture_ms_token(self, text: str) -> Optional[str]:
        for rx in _MS_TOKEN_PATTERNS:
            m = rx.search(text or "")
            if m:
                return m.group("token").strip()
        return None

    def _agent_run(self, routed_message: str, session_id: Optional[str], max_steps: int = 5):
        # Ensure an asyncio event loop exists in this worker thread (e.g., AnyIO/Gradio)
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return self._agent.run(routed_message, session_id=session_id, max_steps=max_steps)

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        # Reset
        if user_message.lower().strip() == "reset session":
            if self._session_id:
                try:
                    self._agent.delete_session(self._session_id)
                except Exception as exc:
                    logging.debug("delete_session failed: %s", exc)
            self._session_id = None
            self._last_domain = None
            self._pending_calendar_msg = None
            yield "Session reset. Ask away!"
            return

        try:
            # 1) If the user pasted an MS token, save it and optionally replay pending ask
            pasted_token = self._maybe_capture_ms_token(user_message or "")
            if pasted_token:
                try:
                    _calendar_toolkit.set_ms_token(pasted_token)
                except Exception:
                    logging.exception("Failed to set MS token")
                    yield TOKEN_PROMPT_HTML
                    return

                if self._pending_calendar_msg:
                    msg_to_rerun = self._pending_calendar_msg
                    self._pending_calendar_msg = None
                    yield "Saved Microsoft Graph token. Checking your calendar now…"
                    routed_message = f"{self._prefix_for_domain('calendar')}{msg_to_rerun}"
                    response = self._agent_run(
                        routed_message,
                        session_id=self._session_id,
                        max_steps=5,
                    )
                    self._session_id = response.session_id
                    self._last_domain = "calendar"
                    text = response.output or str(response)
                    yield text if text.lstrip().startswith("<") else text.replace("\n", "<br>")
                    return

                yield "Saved Microsoft Graph token for this session. Ask your calendar question."
                return

            # 2) Decide domain, allow user override tag
            forced_domain: Optional[str] = None
            m = _ROUTE_TAG_RE.match(user_message or "")
            if m:
                forced_domain = "rag" if m.group(1).upper() == "RAG" else "calendar"
            clean_msg = _ROUTE_TAG_RE.sub("", user_message or "")

            domain = forced_domain or self._infer_domain(clean_msg)

            # If last turn was calendar and this looks like a short follow up, force calendar
            if self._last_domain == "calendar" and _FOLLOWUP_RE.search(clean_msg):
                domain = "calendar"

            # 3) If calendar but token missing, ask for it and remember the ask
            if domain == "calendar" and not _calendar_toolkit.has_ms_token():
                self._pending_calendar_msg = clean_msg
                yield TOKEN_PROMPT_HTML
                return

            # 4) If RAG, call the toolkit directly so we never fall back to general knowledge
            if domain == "rag":
                # pass through the debug flag so rag toolkit prints debug logs when requested
                html = _rag_toolkit.rag(clean_msg, debug=self.debug)
                self._last_domain = "rag"
                yield html if html.lstrip().startswith("<") else html.replace("\n", "<br>")
                return

            # 5) Calendar with token available, run the agent
            routed_message = f"{self._prefix_for_domain(domain)}{clean_msg}"
            response = self._agent_run(
                routed_message,
                session_id=self._session_id,
                max_steps=5,
            )
            self._session_id = response.session_id
            self._last_domain = domain

            text = response.output or str(response)
            yield text if text.lstrip().startswith("<") else text.replace("\n", "<br>")
        except Exception:
            logging.exception("Agent run failed")
            yield "Sorry, something went wrong. Please try again."


def main() -> None:
    parser = argparse.ArgumentParser(description="OCI ADK calendar + RAG chat engine")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--setup", action="store_true", help="Run agent.setup()")
    grp.add_argument("--chat", metavar="PROMPT", help="Single-shot prompt")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        os.environ.setdefault("ADK_LOG_LEVEL", "DEBUG")
        logging.basicConfig(level=logging.DEBUG)

    if args.setup:
        print("Running agent.setup() …")
        calendar_agent.setup()
        print("Setup completed.")
        return

    if args.chat:
        for chunk in ChatEngine(debug=args.debug).chat_stream(args.chat):
            print(chunk)


if __name__ == "__main__":
    main()
