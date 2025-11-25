# src/assistant/share/ics_utils.py
from __future__ import annotations
from datetime import datetime
from typing import Optional


def _format_dt(dt: datetime) -> str:
    """
    Format datetime as UTC in basic ISO format for ICS: YYYYMMDDTHHMMSSZ
    """
    if dt.tzinfo is None:
        # treat as UTC if naive
        return dt.strftime("%Y%m%dT%H%M%SZ")
    return dt.astimezone().strftime("%Y%m%dT%H%M%SZ")


def build_ics_event(
    uid: str,
    title: str,
    start: datetime,
    end: datetime,
    location: str = "",
    description: str = "",
) -> bytes:
    """
    Build a single VEVENT ICS file as bytes.
    """
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//HealthAI//DigitalHealthTwin//EN",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_format_dt(datetime.utcnow())}",
        f"DTSTART:{_format_dt(start)}",
        f"DTEND:{_format_dt(end)}",
        f"SUMMARY:{title}",
    ]

    if location:
        lines.append(f"LOCATION:{location}")

    if description:
        desc = description.replace("\n", "\\n")
        lines.append(f"DESCRIPTION:{desc}")

    lines.extend(
        [
            "END:VEVENT",
            "END:VCALENDAR",
            "",
        ]
    )

    return "\r\n".join(lines).encode("utf-8")
