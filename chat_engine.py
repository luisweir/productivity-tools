#!/usr/bin/env python3
"""
chat_engine.py – calendar-aware + RAG-aware chat engine using OCI ADK.
- Auto-routes non-calendar intents to RAG by default.
- Keeps calendar flows intact (timeframe gating, follow-ups like "with who?" stay on calendar).
- Uses LoadConfig for endpoint/profile/region (same as mic_summary.py).
"""

from __future__ import annotations

import os
import re
import argparse
import logging
from typing import Generator, Optional

from oci.addons.adk import Agent, AgentClient

from calendar_toolkit import CalendarToolkit
from rag_toolkit import RagToolkit
from load_config import LoadConfig  # same pattern as mic_summary.py


# ─────────────────────────────────────────────────────────────
# Router regex: what counts as "calendar intent"?
# (Deliberately broad; tune as you wish.)
_CALENDAR_RE = re.compile(
    r"\b("
    r"calendar|schedule|meeting|meetings|event|events|invite|attendees?|"
    r"busy|free|availability|when|what time|where|room|today|tomorrow|"
    r"this week|next week|next \d+\s*(days?|hours?)"
    r")\b",
    re.IGNORECASE,
)

# short follow-ups that should stay on calendar after a calendar turn
_FOLLOWUP_RE = re.compile(
    r"\b(what time|when exactly|how long|how far|how many minutes|where is it|"
    r"where|with who|attendees?|link|join|location)\b",
    re.IGNORECASE,
)

# user-supplied route tag, which we read then strip
_ROUTE_TAG_RE = re.compile(r"^\s*\[ROUTE:(RAG|CAL)\]\s*", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────
# Build agent (same LoadConfig pattern you had)
def build_agent() -> Agent:
    properties = LoadConfig()
    profile_name = properties.getDefaultProfile()
    agent_endpoint_ocid = properties.getAgentEndpointOcid()
    oci_region = properties.getAgentRegion()

    if not agent_endpoint_ocid or not agent_endpoint_ocid.startswith("ocid1.genaiagentendpoint."):
        raise ValueError(
            "LoadConfig().getAgentEndpointOcid() must return a valid agent-endpoint OCID."
        )

    client = AgentClient(
        auth_type="api_key",
        profile=profile_name,
        region=oci_region,
    )

    # Router-aware instructions with explicit rules
    instructions = (
        """<role>
You are a multi-tool assistant with two capabilities:
1) Calendar assistant (timeframe-gated Microsoft 365 reads).
2) RAG assistant (answer general questions using the rag toolkit).
</role>

<routing-protocol>
- If the user message begins with "[ROUTE:RAG]" → immediately use the RAG toolkit to answer. Do NOT call calendar tools for that turn.
- If the user message begins with "[ROUTE:CAL]" → follow the calendar flow (see below). Do NOT call the RAG toolkit for that turn.
- Otherwise (no prefix) → if the query is clearly about calendar or scheduling, use the calendar flow. For anything else, prefer the RAG toolkit.
</routing-protocol>

<calendar-flow>
- Never fetch calendar without a valid time window (max 7 days). If missing, ask a concise follow-up to get a window (e.g., “today”, “tomorrow”, “next 3 days”, or YYYY-MM-DD..YYYY-MM-DD).
- Call resolve_timeframe(timeframe), then fetch_calendar_events(start_datetime, end_datetime).
- The toolkit auto-detects timezone. Do NOT ask the user for timezone. Present times with the timezone included in tool output.
- For short follow-ups like “with who?”, “where is it?”, “what time exactly?”, or “how long from now?”, assume they refer to the last calendar results and answer directly from those results. If an event object contains "startsInMinutes", use it to answer “how long” questions.
</calendar-flow>

<rag-flow>
- For non-calendar questions, call the RAG toolkit once with the user question and answer with that tool's output.
- Do not answer from your own knowledge for non-calendar turns.
</rag-flow>

<format>
Return clean, simple HTML (no CSS/JS). Use concise paragraphs and lists.
</format>"""
    )

    # Ordering RAG first biases the model toward it for non-calendar turns.
    return Agent(
        client=client,
        agent_endpoint_id=agent_endpoint_ocid,
        instructions=instructions,
        tools=[RagToolkit(), CalendarToolkit()],
    )


# Build once, reuse
calendar_agent: Agent = build_agent()


# ─────────────────────────────────────────────────────────────
#  Chat engine with a lightweight pre-router
# ─────────────────────────────────────────────────────────────
class ChatEngine:
    def __init__(self, debug: bool = False):
        self._agent = calendar_agent
        self._session_id: Optional[str] = None
        self._last_domain: Optional[str] = None  # 'calendar' | 'rag'
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    def _infer_domain(self, text: str) -> str:
        """Heuristic router:
        - If message is short/elliptical and we have a last domain → reuse it.
        - Else if it matches calendar cues → 'calendar'
        - Else → 'rag'
        """
        t = (text or "").strip()
        if len(t) <= 24 and self._last_domain:
            return self._last_domain
        if _CALENDAR_RE.search(t):
            return "calendar"
        return "rag"

    def _prefix_for_domain(self, domain: str) -> str:
        if domain == "calendar":
            return "[ROUTE:CAL] "
        return "[ROUTE:RAG] "

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        # Manual reset
        if user_message.lower().strip() == "reset session":
            if self._session_id:
                try:
                    self._agent.delete_session(self._session_id)
                except Exception as exc:  # noqa: BLE001
                    logging.debug("delete_session failed: %s", exc)
            self._session_id = None
            self._last_domain = None
            yield "Session reset. Ask away!"
            return

        try:
            # detect and strip a user-supplied route tag
            forced_domain: Optional[str] = None
            m = _ROUTE_TAG_RE.match(user_message or "")
            if m:
                forced_domain = "rag" if m.group(1).upper() == "RAG" else "calendar"
            clean_msg = _ROUTE_TAG_RE.sub("", user_message or "")

            # decide route
            domain = forced_domain or self._infer_domain(clean_msg)

            # if last turn was calendar and this looks like a short follow up, force calendar
            if self._last_domain == "calendar" and _FOLLOWUP_RE.search(clean_msg):
                domain = "calendar"

            # if RAG, call the toolkit directly to ensure RAG is used
            if domain == "rag":
                html = RagToolkit().rag(clean_msg)
                self._last_domain = "rag"
                yield html if html.lstrip().startswith("<") else html.replace("\n", "<br>")
                return

            # otherwise use the agent with calendar tools
            routed_message = f"{self._prefix_for_domain(domain)}{clean_msg}"
            response = self._agent.run(
                routed_message,
                session_id=self._session_id,
                max_steps=5,
            )
            self._session_id = response.session_id
            self._last_domain = domain  # remember for short follow-ups

            text = response.output or str(response)
            stripped = text.lstrip()
            # If the agent already produced HTML, do not inject <br>
            if stripped.startswith("<") or "<html" in stripped[:200].lower():
                yield text
            else:
                yield text.replace("\n", "<br>")
        except Exception:
            logging.exception("Agent run failed")
            yield "Sorry, something went wrong. Please try again."


# ─────────────────────────────────────────────────────────────
#  CLI – same as before
# ─────────────────────────────────────────────────────────────
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
