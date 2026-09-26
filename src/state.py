from typing import TypedDict


class ResearchState(TypedDict):
    topic: str

    research_plan: list[str]

    research_results: list[str]

    analysis: str

    draft_report: str

    critique: str

    review_status: str

    iteration: int

    approved: bool