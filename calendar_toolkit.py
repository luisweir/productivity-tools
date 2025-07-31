from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from tzlocal import get_localzone_name
import requests
from zoneinfo import ZoneInfo  # for localised time maths
from oci.addons.adk import Toolkit, tool

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
MAX_DAYS = 7


# -------------------- Time zone helpers & mappings --------------------

# Common IANA → Windows time zone mappings (extend as needed)
_IANA_TO_WINDOWS = {
    "UTC": "UTC",
    "Etc/UTC": "UTC",
    "Europe/London": "GMT Standard Time",
    "Europe/Dublin": "GMT Standard Time",
    "Europe/Paris": "Romance Standard Time",
    "Europe/Berlin": "W. Europe Standard Time",
    "Europe/Madrid": "Romance Standard Time",
    "Europe/Rome": "W. Europe Standard Time",
    "America/New_York": "Eastern Standard Time",
    "America/Chicago": "Central Standard Time",
    "America/Denver": "Mountain Standard Time",
    "America/Phoenix": "US Mountain Standard Time",
    "America/Los_Angeles": "Pacific Standard Time",
    "America/Vancouver": "Pacific Standard Time",
    "America/Toronto": "Eastern Standard Time",
    "Asia/Kolkata": "India Standard Time",
    "Asia/Tokyo": "Tokyo Standard Time",
    "Asia/Shanghai": "China Standard Time",
    "Australia/Sydney": "AUS Eastern Standard Time",
}

# Reverse map for Windows → IANA
_WINDOWS_TO_IANA = {v: k for k, v in _IANA_TO_WINDOWS.items()}


def _to_windows_tz(tz_name: Optional[str]) -> Optional[str]:
    """
    Normalise a time zone to a Windows time zone ID for Microsoft Graph's
    'Prefer: outlook.timezone="..."' header. Accepts either Windows or IANA names.
    """
    if not tz_name:
        return None
    if tz_name in _IANA_TO_WINDOWS.values():
        return tz_name
    return _IANA_TO_WINDOWS.get(tz_name, None)


def _alias_timezone(text: Optional[str]) -> Optional[str]:
    """
    Map user/common aliases to IANA names (then we convert to Windows).
    We intentionally map 'gmt' to Europe/London to preserve DST behaviour.
    """
    if not text:
        return None
    t = text.strip().lower()
    aliases = {
        "uk": "Europe/London",
        "gb": "Europe/London",
        "britain": "Europe/London",
        "england": "Europe/London",
        "london": "Europe/London",
        "bst": "Europe/London",
        "gmt": "Europe/London",
        "utc": "UTC",
    }
    return aliases.get(t, text)


# -------------------- Toolkit --------------------

class CalendarToolkit(Toolkit):
    def __init__(self) -> None:
        super().__init__(name="CalendarToolkit")

    # -------------------- Helper: discover timezone automatically --------------------

    # NOTE: intentionally no caching here
    def _get_mailbox_timezone(self) -> Optional[str]:
        """
        Returns the user's mailbox timezone from Microsoft Graph as a Windows tz id,
        e.g., 'GMT Standard Time'. If not available, returns None.
        """
        token = os.getenv("MS_TOKEN")
        if not token:
            return None

        url = "https://graph.microsoft.com/v1.0/me/mailboxSettings?$select=timeZone"
        try:
            resp = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            data = resp.json() or {}
            tz = (data.get("timeZone") or "").strip()
            return tz or None
        except Exception:
            return None

    def _auto_timezone(self) -> str:
        """
        Determine the best timezone to use for calendar reads (Windows ID preferred):
          1) Outlook mailbox setting (Windows tz id),
          2) Local machine timezone (IANA → Windows),
          3) 'UTC'.
        """
        mailbox_tz = self._get_mailbox_timezone()
        if mailbox_tz:
            return mailbox_tz

        if get_localzone_name:
            try:
                local_iana = get_localzone_name()
                win = _to_windows_tz(local_iana)
                if win:
                    return win
            except Exception:
                pass

        return "UTC"

    # -------------------- Tools --------------------

    @tool
    def resolve_timeframe(
        self,
        timeframe: str,
        now_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse a natural-language timeframe and return a UTC [start, end] window.
        Supported:
          - 'today'
          - 'tomorrow'
          - 'next N days'  (N ∈ [1..7])
          - 'YYYY-MM-DD to YYYY-MM-DD'  (inclusive range, must be ≤ 7 days)
        Output keys:
          - start_datetime (UTC, ISO_FORMAT)
          - end_datetime   (UTC, ISO_FORMAT)
        """
        tf = (timeframe or "").strip().lower()
        if not tf:
            raise ValueError("timeframe is required.")

        # Establish 'now' in UTC
        if now_utc:
            try:
                now = datetime.strptime(now_utc, ISO_FORMAT).replace(tzinfo=timezone.utc)
            except Exception as exc:
                raise ValueError(f"Invalid now_utc format. Use {ISO_FORMAT}.") from exc
        else:
            now = datetime.now(timezone.utc)

        # Helpers to floor/ceil to day boundaries in UTC
        def day_start(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        def day_end(dt: datetime) -> datetime:
            return dt.astimezone(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)

        # 1) Explicit date range: YYYY-MM-DD to YYYY-MM-DD
        m = re.match(r"^\s*(\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(\d{4}-\d{2}-\d{2})\s*$", tf)
        if m:
            d1 = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            d2 = datetime.strptime(m.group(2), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if d2 < d1:
                raise ValueError("End date must be after or equal to start date.")
            if (d2 - d1) > timedelta(days=MAX_DAYS - 1):
                raise ValueError(f"Time frame cannot exceed {MAX_DAYS} days.")
            start_dt = day_start(d1)
            end_dt = day_end(d2)
            return {
                "start_datetime": start_dt.strftime(ISO_FORMAT),
                "end_datetime": end_dt.strftime(ISO_FORMAT),
            }

        # 2) today
        if tf in {"today"}:
            start_dt = day_start(now)
            end_dt = day_end(now)
            return {
                "start_datetime": start_dt.strftime(ISO_FORMAT),
                "end_datetime": end_dt.strftime(ISO_FORMAT),
            }

        # 3) tomorrow
        if tf in {"tomorrow"}:
            tmr = now + timedelta(days=1)
            start_dt = day_start(tmr)
            end_dt = day_end(tmr)
            return {
                "start_datetime": start_dt.strftime(ISO_FORMAT),
                "end_datetime": end_dt.strftime(ISO_FORMAT),
            }

        # 4) next N days
        m = re.match(r"^\s*next\s+(\d{1,2})\s+days?\s*$", tf)
        if m:
            n = int(m.group(1))
            if n < 1:
                raise ValueError("N must be at least 1.")
            if n > MAX_DAYS:
                raise ValueError(f"Time frame cannot exceed {MAX_DAYS} days.")
            start_dt = now  # from now
            end_dt = day_end(now + timedelta(days=n - 1))
            return {
                "start_datetime": start_dt.strftime(ISO_FORMAT),
                "end_datetime": end_dt.strftime(ISO_FORMAT),
            }

        raise ValueError(
            "Unsupported timeframe. Try 'today', 'tomorrow', 'next 3 days', "
            "or 'YYYY-MM-DD to YYYY-MM-DD' (≤ 7 days)."
        )

    @tool
    def fetch_calendar_events(
        self,
        start_datetime: str,
        end_datetime: str,
        max_items: int = 100,
        timezone: Optional[str] = None,  # manual override if the agent/user provides one
    ) -> List[Dict[str, Any]]:
        """
        Fetch events for the requested window. If `timezone` is omitted, the toolkit
        will automatically use the user's Outlook mailbox timezone (preferred), falling
        back to the host machine timezone, then UTC.

        NOTE: We always send a Windows time zone ID in the Graph 'Prefer' header so
        DST rules are applied correctly (e.g., 'GMT Standard Time' for London).
        """
        # validate window
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

        # Decide the timezone to ask Graph to use
        tz_hint = _alias_timezone(timezone)
        win_tz = _to_windows_tz(tz_hint) if tz_hint else None
        if not win_tz:
            win_tz = self._auto_timezone()

        iana = _WINDOWS_TO_IANA.get(win_tz, "UTC")
        now_local = datetime.now(ZoneInfo(iana))

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            'Prefer': f'outlook.timezone="{win_tz}", outlook.body-content-type="text"',
        }

        base_url = "https://graph.microsoft.com/v1.0/me/calendarView"
        params = {
            "startDateTime": start_datetime,
            "endDateTime": end_datetime,
            "$orderby": "start/dateTime",
            "$top": 50,
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
            payload = resp.json() or {}
            events.extend(payload.get("value", []))
            url = payload.get("@odata.nextLink")

        compacted: List[Dict[str, Any]] = []
        for e in events[:max_items]:
            attendees: List[str] = []
            for a in (e.get("attendees") or [])[:5]:
                em = a.get("emailAddress") or {}
                nm = em.get("name") or em.get("address")
                if nm:
                    attendees.append(nm)

            om = e.get("onlineMeeting") or {}
            join_url = om.get("joinUrl") or e.get("onlineMeetingUrl")

            start_obj = e.get("start") or {}
            end_obj = e.get("end") or {}

            # compute startsInMinutes using the same local zone
            start_local_str = start_obj.get("dateTime") or ""
            start_local_dt = None
            try:
                # Graph often returns "YYYY-MM-DDTHH:MM:SS.fffffff"
                # Keep up to seconds and attach the zone
                start_local_dt = datetime.fromisoformat(start_local_str[:19]).replace(tzinfo=ZoneInfo(iana))
            except Exception:
                start_local_dt = None

            starts_in_minutes = None
            if start_local_dt:
                delta_sec = (start_local_dt - now_local).total_seconds()
                starts_in_minutes = int(delta_sec // 60)

            compacted.append({
                "id": e.get("id"),
                "subject": e.get("subject"),
                # Graph will return these normalised to the requested win_tz
                "start": start_obj.get("dateTime"),
                "startTimeZone": start_obj.get("timeZone"),
                "end": end_obj.get("dateTime"),
                "endTimeZone": end_obj.get("timeZone"),
                "isAllDay": e.get("isAllDay"),
                "isCancelled": e.get("isCancelled"),
                "showAs": e.get("showAs"),
                "location": (e.get("location") or {}).get("displayName"),
                "webLink": e.get("webLink"),
                "onlineMeetingUrl": join_url,
                "organizer": ((e.get("organizer") or {}).get("emailAddress") or {}).get("name"),
                "attendees": attendees,
                "startsInMinutes": starts_in_minutes,  # helpful for “how long from now?”
            })

        return compacted
