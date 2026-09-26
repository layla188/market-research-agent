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
        note = {
            "claim": claim,
            "source": source,
            "topic": topic,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
        }

        self.notes.append(note)

        return {
            "status": "saved",
            "note": note,
        }

    def get_notes(self, topic: str | None = None) -> list[dict]:
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