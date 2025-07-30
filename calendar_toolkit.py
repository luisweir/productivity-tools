"""
calendar_toolkit.py - Calendar toolkit for chat engine using OCI ADK.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from oci.addons.adk import Toolkit, tool

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MAX_DAYS = 7


class CalendarToolkit(Toolkit):
    @tool
    def resolve_timeframe(
        self,
        timeframe: str,
        now_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse a human time frame into UTC start and end datetimes.
        Allowed examples: "today", "tomorrow", "next 3 days",
        "2025-07-30", "2025-07-30 to 2025-08-01", "this week", "next week".
        Enforces a maximum of 7 days. Returns an error if over the limit.
        """
        now = datetime.strptime(now_utc, ISO_FORMAT) if now_utc else datetime.now(timezone.utc)
        today = now.date()

        def iso_start(dt: datetime) -> str:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc).strftime(ISO_FORMAT)

        def iso_end(dt: datetime) -> str:
            # exclusive end at 00:00 of the day after
            next_day = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc) + timedelta(days=1)
            return next_day.strftime(ISO_FORMAT)

        tf = timeframe.strip().lower()

        start_date = None
        end_date_inclusive = None
        label = timeframe

        # simple phrases
        if tf == "today":
            start_date = today
            end_date_inclusive = today
            label = "today"
        elif tf == "tomorrow":
            start_date = today + timedelta(days=1)
            end_date_inclusive = start_date
            label = "tomorrow"
        elif tf in {"this week", "current week"}:
            # Monday to Sunday of current ISO week
            start_date = today - timedelta(days=today.weekday())
            end_date_inclusive = start_date + timedelta(days=6)
            label = "this week"
        elif tf in {"next week"}:
            start_date = today - timedelta(days=today.weekday()) + timedelta(days=7)
            end_date_inclusive = start_date + timedelta(days=6)
            label = "next week"
        else:
            # next N days
            m = re.fullmatch(r"next\s+(\d{1,2})\s+days?", tf)
            if m:
                n = int(m.group(1))
                if n < 1:
                    return {"error": "Time frame must be at least 1 day.", "max_days": MAX_DAYS}
                start_date = today
                end_date_inclusive = today + timedelta(days=n - 1)
                label = f"next {n} days"

            # YYYY-MM-DD
            if start_date is None:
                m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})", tf)
                if m:
                    d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                    start_date = d
                    end_date_inclusive = d
                    label = m.group(1)

            # YYYY-MM-DD to YYYY-MM-DD
            if start_date is None:
                m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*(to|-)\s*(\d{4}-\d{2}-\d{2})", tf)
                if m:
                    d1 = datetime.strptime(m.group(1), "%Y-%m-%d").date()
                    d2 = datetime.strptime(m.group(3), "%Y-%m-%d").date()
                    if d2 < d1:
                        d1, d2 = d2, d1
                    start_date = d1
                    end_date_inclusive = d2
                    label = f"{d1.isoformat()} to {d2.isoformat()}"

        if start_date is None or end_date_inclusive is None:
            return {"error": "Could not parse time frame. Use phrases like 'today', 'tomorrow', 'next 3 days', or ISO dates like '2025-08-01 to 2025-08-03'."}

        days = (end_date_inclusive - start_date).days + 1
        if days > MAX_DAYS:
            return {"error": f"Time frame cannot exceed {MAX_DAYS} days.", "max_days": MAX_DAYS, "parsed_days": days, "label": label}

        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_dt_exclusive = datetime.combine(end_date_inclusive, datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)

        return {
            "start_datetime": start_dt.strftime(ISO_FORMAT),
            "end_datetime": end_dt_exclusive.strftime(ISO_FORMAT),
            "days": days,
            "label": label,
        }

    @tool
    def fetch_calendar_events(
        self,
        start_datetime: str,
        end_datetime: str,
        max_items: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Return a compact list of events for the requested window in UTC.
        Enforces a maximum window of 7 days.
        """
        # validate window length
        try:
            start = datetime.strptime(start_datetime, ISO_FORMAT)
            end = datetime.strptime(end_datetime, ISO_FORMAT)
        except Exception as exc:
            raise ValueError(f"Invalid datetime format. Use {ISO_FORMAT}.") from exc

        if end <= start:
            raise ValueError("end_datetime must be after start_datetime.")

        if (end - start) > timedelta(days=MAX_DAYS):
            raise ValueError(f"Time frame cannot exceed {MAX_DAYS} days.")

        token = os.getenv("MS_TOKEN")
        if not token:
            raise EnvironmentError("MS_TOKEN environment variable not set")

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            # Keep everything in UTC and avoid large HTML bodies
            "Prefer": 'outlook.timezone="UTC", outlook.body-content-type="text"',
        }

        base_url = "https://graph.microsoft.com/v1.0/me/calendarView"
        params = {
            "startDateTime": start_datetime,
            "endDateTime": end_datetime,
            "$orderby": "start/dateTime",
            "$top": 50,  # paging handled below
            "$select": (
                "id,subject,start,end,location,showAs,isAllDay,isCancelled,"
                "organizer,attendees,webLink,onlineMeeting,onlineMeetingUrl"
            ),
        }

        events: List[Dict[str, Any]] = []
        url = f"{base_url}?{urlencode(params)}"
        while url and len(events) < max_items:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            batch = payload.get("value", [])
            events.extend(batch)
            url = payload.get("@odata.nextLink")

        # compact further, trim attendees to names only and limit length
        compacted: List[Dict[str, Any]] = []
        for e in events[:max_items]:
            attendees = []
            for a in (e.get("attendees") or [])[:5]:
                nm = ((a.get("emailAddress") or {}).get("name")) or ((a.get("emailAddress") or {}).get("address"))
                if nm:
                    attendees.append(nm)

            # Prefer onlineMeeting.joinUrl when available; fall back to legacy onlineMeetingUrl
            join_url = None
            om = e.get("onlineMeeting") or {}
            if isinstance(om, dict):
                join_url = om.get("joinUrl")
            join_url = join_url or e.get("onlineMeetingUrl")

            compacted.append({
                "id": e.get("id"),
                "subject": e.get("subject"),
                "start": (e.get("start") or {}).get("dateTime"),
                "end": (e.get("end") or {}).get("dateTime"),
                "isAllDay": e.get("isAllDay"),
                "isCancelled": e.get("isCancelled"),
                "showAs": e.get("showAs"),
                "location": (e.get("location") or {}).get("displayName"),
                "webLink": e.get("webLink"),
                "onlineMeetingUrl": join_url,
                "organizer": ((e.get("organizer") or {}).get("emailAddress") or {}).get("name"),
                "attendees": attendees,
            })

        return compacted
