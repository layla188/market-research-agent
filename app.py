"""
Streamlit interface for the Autonomous Market Research Agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import streamlit as st
from langgraph.types import Command

from src.graph import research_graph


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Autonomous Market Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

DEFAULT_SESSION_VALUES = {
    "thread_id": None,
    "topic": "",
    "result": None,
    "phase": "idle",  # idle | approval | complete | error
    "error": "",
    "pending_report": "",
    "final_report": "",
    "interrupt_payload": None,
}


def initialize_session() -> None:
    for key, value in DEFAULT_SESSION_VALUES.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.thread_id is None:
        st.session_state.thread_id = str(uuid.uuid4())


def reset_session() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.topic = ""
    st.session_state.result = None
    st.session_state.phase = "idle"
    st.session_state.error = ""
    st.session_state.pending_report = ""
    st.session_state.final_report = ""
    st.session_state.interrupt_payload = None


initialize_session()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def build_initial_state(topic: str) -> dict[str, Any]:
    return {
        "topic": topic,
        "research_plan": [],
        "research_results": [],
        "evidence_status": "",
        "research_gaps": [],
        "research_iteration": 0,
        "analysis": "",
        "draft_report": "",
        "critique": "",
        "review_status": "",
        "iteration": 0,
        "approved": False,
        "memory_notes": [],
        "errors": [],
    }


def graph_config() -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id,
        }
    }


def normalize_text(value: Any) -> str:
    """
    Convert model / graph output to displayable text.

    Handles normal strings and common structured content formats returned
    by chat providers.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        parts: list[str] = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                # Common content-block shapes.
                text_value = (
                    item.get("text")
                    or item.get("content")
                    or item.get("value")
                )

                if isinstance(text_value, str):
                    parts.append(text_value)

        if parts:
            return "\n\n".join(part.strip() for part in parts if part.strip())

        return json.dumps(value, ensure_ascii=False, indent=2)

    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            text_value = value.get(key)

            if isinstance(text_value, str):
                return text_value.strip()

        return json.dumps(value, ensure_ascii=False, indent=2)

    return str(value).strip()


def looks_like_report(text: str) -> bool:
    """
    Reject obviously incomplete provider output such as tiny metadata strings.

    A real generated report in this project is much longer than a few words.
    """
    clean = normalize_text(text)

    if len(clean) < 300:
        return False

    report_markers = (
        "Executive Summary",
        "Market Overview",
        "Competitive",
        "Sources",
    )

    return any(marker.lower() in clean.lower() for marker in report_markers)


def get_interrupt_payload(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not result:
        return None

    interrupts = result.get("__interrupt__")

    if not interrupts:
        return None

    first_interrupt = interrupts[0]

    if hasattr(first_interrupt, "value"):
        payload = first_interrupt.value
    elif isinstance(first_interrupt, dict):
        payload = first_interrupt
    else:
        return None

    if isinstance(payload, dict):
        return payload

    return {"message": str(payload)}


def get_checkpoint_state() -> dict[str, Any]:
    """
    Read the canonical checkpointed state from LangGraph.

    This is more reliable after interrupt/resume than depending only on the
    object returned by invoke().
    """
    try:
        snapshot = research_graph.get_state(graph_config())

        if snapshot is None:
            return {}

        values = getattr(snapshot, "values", None)

        if isinstance(values, dict):
            return dict(values)

    except Exception:
        # UI should still be able to use invoke() output if snapshot lookup
        # is unavailable in a particular LangGraph version.
        pass

    return {}


def merge_states(*states: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for state in states:
        if isinstance(state, dict):
            merged.update(state)

    return merged


def choose_report(*candidates: Any) -> str:
    """
    Pick the best valid report candidate.

    Preference is given in the order provided. Tiny metadata-only strings
    such as "User Safety: safe" are ignored.
    """
    normalized = [normalize_text(value) for value in candidates]

    for text in normalized:
        if looks_like_report(text):
            return text

    # If no candidate passes the strong report check, keep the longest text.
    non_empty = [text for text in normalized if text]

    if non_empty:
        return max(non_empty, key=len)

    return ""


# -------------------------------------------------------------------
# Graph execution
# -------------------------------------------------------------------

def run_research(topic: str) -> None:
    try:
        st.session_state.error = ""
        st.session_state.topic = topic
        st.session_state.pending_report = ""
        st.session_state.final_report = ""
        st.session_state.interrupt_payload = None

        invoke_result = research_graph.invoke(
            build_initial_state(topic),
            config=graph_config(),
        )

        checkpoint_state = get_checkpoint_state()

        result = merge_states(
            invoke_result,
            checkpoint_state,
        )

        payload = get_interrupt_payload(invoke_result)

        if payload:
            # Preserve exactly the report that the human is asked to review.
            # This prevents the report from disappearing after graph resume.
            reviewed_report = choose_report(
                payload.get("report"),
                result.get("draft_report"),
                invoke_result.get("draft_report")
                if isinstance(invoke_result, dict)
                else "",
            )

            st.session_state.pending_report = reviewed_report
            st.session_state.interrupt_payload = payload
            st.session_state.result = result
            st.session_state.phase = "approval"

        else:
            final_report = choose_report(
                result.get("draft_report"),
                invoke_result.get("draft_report")
                if isinstance(invoke_result, dict)
                else "",
            )

            st.session_state.final_report = final_report
            st.session_state.result = result
            st.session_state.phase = "complete"

    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.phase = "error"


def resume_after_human_decision(decision: str) -> None:
    """
    Resume the interrupted graph.

    Important:
    The report shown during human approval is persisted separately in
    session state. After resume, the app also reads the canonical LangGraph
    checkpoint and merges it with the resume result.
    """
    try:
        st.session_state.error = ""

        before_resume = st.session_state.result or {}
        reviewed_report = st.session_state.pending_report

        resume_result = research_graph.invoke(
            Command(resume=decision),
            config=graph_config(),
        )

        checkpoint_state = get_checkpoint_state()

        final_state = merge_states(
            before_resume,
            resume_result,
            checkpoint_state,
        )

        # Human approval does not need to rewrite the report. The safest
        # display/download value is therefore the report that was actually
        # shown to the user before approval.
        final_report = choose_report(
            reviewed_report,
            final_state.get("draft_report"),
            before_resume.get("draft_report")
            if isinstance(before_resume, dict)
            else "",
        )

        # Keep the valid report in the UI state even if a provider returned
        # a short metadata string in draft_report on a later step.
        if final_report:
            final_state["draft_report"] = final_report

        st.session_state.final_report = final_report
        st.session_state.result = final_state
        st.session_state.interrupt_payload = None
        st.session_state.phase = "complete"

    except Exception as exc:
        st.session_state.error = str(exc)
        st.session_state.phase = "error"


# -------------------------------------------------------------------
# Rendering
# -------------------------------------------------------------------

def render_status(result: dict[str, Any]) -> None:
    st.subheader("Workflow status")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Evidence",
        result.get("evidence_status") or "—",
    )

    col2.metric(
        "Research loop",
        result.get("research_iteration", 0),
    )

    col3.metric(
        "Report revisions",
        result.get("iteration", 0),
    )

    col4.metric(
        "Reviewer status",
        result.get("review_status") or "—",
    )

    gaps = result.get("research_gaps", [])
    errors = result.get("errors", [])

    if gaps:
        with st.expander(f"Research gaps ({len(gaps)})"):
            for gap in gaps:
                st.write(f"• {gap}")

    if errors:
        with st.expander(
            f"Runtime errors ({len(errors)})",
            expanded=True,
        ):
            for error in errors:
                st.error(error)


def render_report(
    report: Any,
    heading: str = "Research report",
) -> None:
    report_text = normalize_text(report)

    st.subheader(heading)

    if not report_text:
        st.warning("No report text is available.")
        return

    if not looks_like_report(report_text):
        st.warning(
            "The workflow returned unusually short report content. "
            "Open 'Final workflow state' below to inspect the raw state."
        )

    with st.container(border=True):
        st.markdown(report_text)

    filename_topic = (
        st.session_state.topic.lower()
        .replace(" ", "-")
        .replace("/", "-")
    )[:60]

    st.download_button(
        label="Download report as Markdown",
        data=report_text.encode("utf-8"),
        file_name=f"{filename_topic or 'market-research-report'}.md",
        mime="text/markdown",
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.title("Research Agent")
    st.caption(
        "Autonomous market research with tools, memory, "
        "guardrails, and human approval."
    )

    st.divider()

    st.markdown(
        """
**Workflow**

1. Plan
2. Research
3. Evaluate evidence
4. Analyze
5. Write
6. Review / revise
7. Human approval
"""
    )

    st.divider()

    st.markdown(
        """
**Research tools**

- Web search
- Market metrics calculator
- Research memory
"""
    )

    st.divider()

    st.caption(
        f"Thread: `{st.session_state.thread_id[:8]}…`"
    )

    if st.button(
        "Start a new research session",
        use_container_width=True,
    ):
        reset_session()
        st.rerun()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

st.title("Autonomous Market Research Agent")
st.caption(
    "Generate an evidence-based market research report through "
    "a traced, multi-step LangGraph workflow."
)

st.divider()


# -------------------------------------------------------------------
# Input
# -------------------------------------------------------------------

if st.session_state.phase == "idle":
    st.subheader("Research objective")

    with st.form("research_form"):
        topic = st.text_input(
            "Market research topic",
            placeholder=(
                "e.g. AI-powered customer support software market"
            ),
        )

        submitted = st.form_submit_button(
            "Run research",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        clean_topic = topic.strip()

        if not clean_topic:
            st.warning("Enter a market research topic first.")

        else:
            with st.status(
                "Running the autonomous research workflow...",
                expanded=True,
            ) as status:
                st.write("Planning the research scope")
                st.write("Collecting and evaluating evidence")
                st.write("Generating and reviewing the report")

                run_research(clean_topic)

                if st.session_state.phase == "error":
                    status.update(
                        label="Workflow failed",
                        state="error",
                        expanded=True,
                    )

                elif st.session_state.phase == "approval":
                    status.update(
                        label=(
                            "Research completed — "
                            "human approval required"
                        ),
                        state="complete",
                        expanded=False,
                    )

                else:
                    status.update(
                        label="Workflow completed",
                        state="complete",
                        expanded=False,
                    )

            st.rerun()


# -------------------------------------------------------------------
# Human approval
# -------------------------------------------------------------------

elif st.session_state.phase == "approval":
    result = st.session_state.result or {}
    payload = st.session_state.interrupt_payload or {}

    st.info(
        payload.get(
            "message",
            "The workflow is paused and waiting for human approval.",
        )
    )

    render_status(result)

    report = choose_report(
        st.session_state.pending_report,
        payload.get("report"),
        result.get("draft_report"),
    )

    render_report(
        report,
        heading="Report awaiting approval",
    )

    st.subheader("Human decision")

    st.caption(
        "Approve to finalize this workflow, or reject to "
        "finish the run without approval."
    )

    approve_col, reject_col = st.columns(2)

    with approve_col:
        if st.button(
            "Approve report",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Resuming workflow..."):
                resume_after_human_decision("approve")

            st.rerun()

    with reject_col:
        if st.button(
            "Reject report",
            use_container_width=True,
        ):
            with st.spinner("Resuming workflow..."):
                resume_after_human_decision("reject")

            st.rerun()


# -------------------------------------------------------------------
# Completed workflow
# -------------------------------------------------------------------

elif st.session_state.phase == "complete":
    result = st.session_state.result or {}

    approved = bool(result.get("approved"))

    if approved:
        st.success(
            "Workflow completed and the report was approved."
        )
    else:
        st.warning(
            "Workflow completed without human approval."
        )

    render_status(result)

    final_report = choose_report(
        st.session_state.final_report,
        st.session_state.pending_report,
        result.get("draft_report"),
    )

    render_report(
        final_report,
        heading="Final market research report",
    )

    with st.expander("Research memory used in this run"):
        notes = result.get("memory_notes", [])

        if not notes:
            st.caption(
                "No research-memory entries are present "
                "in the final state."
            )
        else:
            for index, note in enumerate(
                notes,
                start=1,
            ):
                st.markdown(f"**Note {index}**")
                st.write(note)

    with st.expander("Final workflow state"):
        safe_state = {
            key: value
            for key, value in result.items()
            if key != "__interrupt__"
        }

        st.json(safe_state)

    st.divider()

    if st.button(
        "Research another market",
        type="primary",
        use_container_width=True,
    ):
        reset_session()
        st.rerun()


# -------------------------------------------------------------------
# Error
# -------------------------------------------------------------------

elif st.session_state.phase == "error":
    st.error("The workflow could not complete.")

    if st.session_state.error:
        st.code(st.session_state.error)

    st.caption(
        "Check API keys, model availability, Tavily quota, "
        "and the terminal traceback."
    )

    if st.button(
        "Start a new session",
        type="primary",
        use_container_width=True,
    ):
        reset_session()
        st.rerun()