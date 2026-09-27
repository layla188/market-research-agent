from datetime import datetime
from typing import Any


class ResearchMemory:

    def __init__(self):
        self.notes: list[dict[str, Any]] = []

    def save_note(
        self,
        claim: str,
        source: str,
        topic: str,
        date: str | None = None,
    ) -> dict:

        if not claim or not claim.strip():
            raise ValueError(
                "Research claim cannot be empty."
            )

        if not source or not source.strip():
            raise ValueError(
                "Research source cannot be empty."
            )

        if not topic or not topic.strip():
            raise ValueError(
                "Research topic cannot be empty."
            )

        note = {
            "claim": claim.strip(),
            "source": source.strip(),
            "topic": topic.strip(),
            "date": (
                date
                or datetime.now().strftime("%Y-%m-%d")
            ),
        }

        self.notes.append(note)

        return {
            "status": "saved",
            "note": note,
        }

    def get_notes(
        self,
        topic: str | None = None,
    ) -> list[dict]:

        if topic is None:
            return self.notes.copy()

        topic_lower = topic.lower()

        return [
            note
            for note in self.notes
            if topic_lower in note["topic"].lower()
        ]

    def clear(self):
        self.notes.clear()