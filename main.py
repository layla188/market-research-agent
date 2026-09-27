"""
Main entry point for the Market Research Agent.

Runs the complete autonomous research workflow:

Planner
   ↓
Researcher
   ↓
Analyzer
   ↓
Researcher (if evidence is insufficient)
   ↓
Writer
   ↓
Reviewer
   ↓
Revision loop (if needed)
   ↓
Human Approval
   ↓
Final Report
"""

from langgraph.types import Command

from src.graph import research_graph


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_workflow_status(result: dict):
    """Display important workflow information."""

    print_header("WORKFLOW STATUS")

    evidence_status = result.get("evidence_status", "")
    research_iteration = result.get("research_iteration", 0)
    review_status = result.get("review_status", "")
    revision_iteration = result.get("iteration", 0)

    print(f"Research iterations : {research_iteration}")
    print(f"Evidence status     : {evidence_status}")
    print(f"Report revisions    : {revision_iteration}")
    print(f"Review status       : {review_status}")


def print_final_report(result: dict):
    """Display the generated market research report."""

    report = result.get("draft_report", "")

    print_header("FINAL MARKET RESEARCH REPORT")

    if report:
        print(report)
    else:
        print("No report was generated.")


def handle_human_approval(result: dict, config: dict):
    """
    Handle the human approval interrupt.

    The graph pauses before finalizing the report.
    The user can approve or reject the generated report.
    """

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        return result

    interrupt_data = interrupts[0].value

    print_header("HUMAN APPROVAL REQUIRED")

    print(interrupt_data.get("message", "Please review the report."))

    print("\nGenerated report:")
    print("-" * 70)
    print(interrupt_data.get("report", ""))

    while True:
        decision = input(
            "\nEnter your decision (approve/reject): "
        ).strip().lower()

        if decision in {"approve", "reject"}:
            break

        print("Invalid choice. Please enter 'approve' or 'reject'.")

    print(f"\nHuman decision: {decision}")

    # Resume the interrupted graph.
    resumed_result = research_graph.invoke(
        Command(resume=decision),
        config=config,
    )

    return resumed_result


def main():
    print_header("AUTONOMOUS MARKET RESEARCH AGENT")

    print(
        """
This agent will:

1. Plan the research
2. Perform web research
3. Analyze evidence
4. Decide whether more research is needed
5. Generate a market research report
6. Review the report
7. Revise it if necessary
8. Ask for human approval
9. Finalize the report
"""
    )

    topic = input(
        "Enter the market research topic: "
    ).strip()

    if not topic:
        print("\nError: research topic cannot be empty.")
        return

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    initial_state = {
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

    # --------------------------------------------------------
    # Thread configuration
    # --------------------------------------------------------
    #
    # The thread_id allows LangGraph's checkpointer to keep
    # this workflow execution associated with one conversation.
    #
    # This is important because human approval uses interrupt()
    # and later resumes the same graph execution.
    # --------------------------------------------------------

    config = {
        "configurable": {
            "thread_id": "market-research-session-1"
        }
    }

    print_header("STARTING WORKFLOW")

    try:

        # ----------------------------------------------------
        # Run the complete graph
        # ----------------------------------------------------

        result = research_graph.invoke(
            initial_state,
            config=config,
        )

        # ----------------------------------------------------
        # Human approval
        # ----------------------------------------------------

        if result.get("__interrupt__"):

            result = handle_human_approval(
                result,
                config,
            )

        # ----------------------------------------------------
        # Display final workflow information
        # ----------------------------------------------------

        print_workflow_status(result)

        # ----------------------------------------------------
        # Display final report
        # ----------------------------------------------------

        print_final_report(result)

        # ----------------------------------------------------
        # Display errors if any
        # ----------------------------------------------------

        errors = result.get("errors", [])

        if errors:

            print_header("WORKFLOW ERRORS")

            for error in errors:
                print(f"- {error}")

        # ----------------------------------------------------
        # Final approval status
        # ----------------------------------------------------

        approved = result.get("approved", False)

        print_header("FINAL RESULT")

        if approved:
            print("Report approved by human.")
        else:
            print("Report was not approved.")

    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")

    except Exception as e:
        print_header("WORKFLOW FAILED")

        print(f"Error: {e}")


if __name__ == "__main__":
    main()