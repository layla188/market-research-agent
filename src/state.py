from typing import TypedDict


class ResearchState(TypedDict):
    topic: str

    research_plan: list[str]
    research_results: list[str]

    # Used by the analyzer to decide
    # whether more research is needed.
    evidence_status: str
    research_gaps: list[str]
    research_iteration: int

    analysis: str

    draft_report: str
    critique: str
    review_status: str

    iteration: int
    approved: bool

    memory_notes: list[dict]