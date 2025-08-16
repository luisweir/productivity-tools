#!/usr/bin/env python
# ms_calendar_to_csv.py - Fetch Outlook calendar events and save to CSV
#
# Prerequisites:
#   1) Install dependencies:
#      pip install requests
#
#   2) Get a Microsoft Graph OAuth Token:
#      - Go to: https://developer.microsoft.com/en-us/graph/graph-explorer
#      - Sign in (top right corner)
#      - Click on the "Access token" tab
#      - Copy the token and export it as an environment variable:
#        export MS_TOKEN="paste_your_token_here"
#
# Usage:
#   python ms_calendar_to_csv.py --start 2025-08-10T00:00:00Z --end 2025-08-15T23:59:59Z
#
# Output:
#   Generates a CSV file named `calendar_events.csv` with structured calendar data

import os
import csv
import requests
import argparse
from urllib.parse import urlencode

def fetch_calendar_events(start_datetime, end_datetime):
    token = os.getenv("MS_TOKEN")
    if not token:
        raise EnvironmentError("MS_TOKEN environment variable not set.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Prefer": 'outlook.timezone="UTC"'
    }

    base_url = "https://graph.microsoft.com/v1.0/me/calendarview"
    params = {
        "startdatetime": start_datetime,
        "enddatetime": end_datetime
    }
    events = []
    url = f"{base_url}?{urlencode(params)}"

    while url:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch calendar events: {response.text}")
        data = response.json()
        events.extend(data.get("value", []))
        url = data.get("@odata.nextLink")

    return events

def write_to_csv(events, filename="calendar_events.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = [
            "subject", "start", "end", "location", "organizer", "attendees",
            "isOnlineMeeting", "onlineMeetingUrl", "webLink", "bodyPreview"
        ]
        writer.writerow(header)
        for event in events:
            attendees = ", ".join([a["emailAddress"]["address"] for a in event.get("attendees", [])])
            writer.writerow([
                event.get("subject", ""),
                event.get("start", {}).get("dateTime", ""),
                event.get("end", {}).get("dateTime", ""),
                event.get("location", {}).get("displayName", ""),
                event.get("organizer", {}).get("emailAddress", {}).get("address", ""),
                attendees,
                event.get("isOnlineMeeting", False),
                event.get("onlineMeetingUrl", ""),
                event.get("webLink", ""),
                event.get("bodyPreview", "").replace("\n", " ").replace("\r", "")
            ])

def main():
    parser = argparse.ArgumentParser(description="Fetch Outlook calendar events and export to CSV.")
    parser.add_argument("--start", required=True, help="Start datetime in ISO format (e.g., 2025-07-28T00:00:00Z)")
    parser.add_argument("--end", required=True, help="End datetime in ISO format (e.g., 2025-08-01T23:59:59Z)")
    args = parser.parse_args()

    events = fetch_calendar_events(args.start, args.end)
    write_to_csv(events)
    print(f"Exported {len(events)} events to calendar_events.csv")

if __name__ == "__main__":
    main()
