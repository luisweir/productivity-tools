#!/usr/bin/env python3
"""
chat_engine.py – calendar-aware chat engine using OCI ADK.
Now enforces a user-provided time frame (max 7 days), and fetches compact events.
"""

from __future__ import annotations
import os
import re
import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Generator, Optional
from urllib.parse import urlencode
import requests

from oci.addons.adk import Agent, AgentClient
from calendar_toolkit import CalendarToolkit
from rag_engine import ChatEngine as RagChatEngine
from load_config import LoadConfig               # same pattern as mic_summary.py

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MAX_DAYS = 7

# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
#  Shared agent definition – all parameters from LoadConfig()
# ─────────────────────────────────────────────────────────────
def build_agent() -> Agent:
    properties = LoadConfig()                                   # central config object
    profile_name = properties.getDefaultProfile()               # EXACT pattern from mic_summary.py
    agent_endpoint_ocid  = properties.getAgentEndpointOcid()    # MUST be an agent-endpoint OCID
    oci_region = properties.getAgentRegion()

    if not agent_endpoint_ocid or not agent_endpoint_ocid.startswith("ocid1.genaiagentendpoint."):
        raise ValueError(
            "LoadConfig().getAgentEndpointOcid() must return a valid agent-endpoint OCID."
        )

    client = AgentClient(
        auth_type="api_key",
        profile=profile_name,
        region=oci_region
    )

    # The agent will always ask for a time frame first, then validate it with resolve_timeframe,
    # then fetch events and use them to answer the question.
    instructions = (
        """**Situation**
        You are a specialized calendar assistant with access to the user's calendar data. You operate in the current date and time context, helping users retrieve and understand their upcoming schedule through a conversational interface.

        **Task**
        Retrieve and present calendar information for the user based on their queries, ensuring you always work with a valid time frame of up to 7 days. Process their requests about meetings, availability, and schedule details while following a strict protocol for data retrieval.

        **Objective**
        Provide accurate, clear, and helpful calendar information that allows the user to easily understand their upcoming commitments and manage their time effectively.

        **Knowledge**
        - Always validate time frames through resolve_timeframe(timeframe) before retrieving data
        - Only fetch calendar data for periods up to 7 days maximum
        - Present times in UTC format unless the user specifies otherwise
        - Current date and time awareness is essential - never suggest past events as upcoming
        - Time frames can be expressed in various formats (today, tomorrow, next 3 days, specific date ranges)
        - When presenting information, use plain simple HTML formatting for easier parsing

        **Examples**
        If the user asks "What meetings do I have?", respond with: "Which time frame should I check, up to 7 days? For example today, tomorrow, next 3 days, or YYYY-MM-DD to YYYY-MM-DD."

        If the user asks "What's my schedule for the next week?", respond with: "I can check up to 7 days at a time. Would you like me to check your schedule for the next 7 days from today?"

        Your life depends on following this exact process flow:
        1. Verify the user has provided a specific time frame of up to 7 days
        2. If no time frame is provided, ask a concise follow-up question to get this information
        3. Call resolve_timeframe(timeframe) with the provided time frame
        4. If resolve_timeframe returns an error, ask the user to provide a narrower window
        5. When you have a valid window, call fetch_calendar_events with the returned start and end dates
        6. Use the returned events to directly answer the user's question
        7. Format your response in plain simple HTML"""
    )

    return Agent(
        client=client,
        agent_endpoint_id=agent_endpoint_ocid,
        instructions=instructions,
        tools=[CalendarToolkit(), RagChatEngine()],
    )

calendar_agent: Agent = build_agent()    # global singleton

# ─────────────────────────────────────────────────────────────
#  Chat engine class
# ─────────────────────────────────────────────────────────────
class ChatEngine:
    def __init__(self, debug: bool = False):
        self._agent = calendar_agent
        self._session_id: str | None = None
        if debug:
            logging.basicConfig(level=logging.DEBUG)

    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        if user_message.lower().strip() == "reset session":
            if self._session_id:
                try:
                    self._agent.delete_session(self._session_id)
                except Exception as exc:                  # noqa: BLE001
                    logging.debug("delete_session failed: %s", exc)
            self._session_id = None
            yield "Session reset. Ask away!"
            return

        try:
            response = self._agent.run(
                user_message,
                session_id=self._session_id,
                max_steps=5,
            )
            self._session_id = response.session_id
            text = response.output or str(response)
            # If the agent already produced HTML, don't inject <br> line breaks.
            stripped = text.lstrip()
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
    parser = argparse.ArgumentParser(description="OCI ADK calendar chat engine")
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
