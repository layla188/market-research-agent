from src.graph import research_graph
from src.tools import (
    calculate_market_metrics,
    save_research_note,
    get_research_notes,
    research_memory,
)


def test_graph():
    print("\n[1] Checking graph...")

    print(f"Graph object: {type(research_graph).__name__}")

    assert research_graph is not None

    print("Graph compiled: OK")


def test_tools():
    print("\n[2] Checking tools...")

    assert calculate_market_metrics is not None
    assert save_research_note is not None
    assert get_research_notes is not None

    print("✓ calculate_market_metrics")
    print("✓ save_research_note")
    print("✓ get_research_notes")

    print("Tools: OK")


def test_calculator():
    print("\n[3] Checking calculator...")

    result = calculate_market_metrics.invoke(
        {
            "metric": "cagr",
            "initial_value": 100,
            "final_value": 200,
            "years": 5,
        }
    )

    print(f"Calculator: {result}")

    assert "14.87%" in result

    print("Calculator: OK")


def test_memory():
    print("\n[4] Checking memory...")

    # Clear old test data first
    research_memory.clear()

    save_result = save_research_note.invoke(
        {
            "claim": "Test research claim.",
            "source": "Test Source",
            "topic": "structure-test",
            "date": "2026-09-27",
        }
    )

    print(f"Save: {save_result}")

    assert "saved" in save_result

    retrieve_result = get_research_notes.invoke(
        {
            "topic": "structure-test"
        }
    )

    print(f"Retrieve: {retrieve_result}")

    assert "Test research claim." in retrieve_result
    assert "Test Source" in retrieve_result

    print("Memory: OK")


def test_empty_memory_validation():
    print("\n[5] Checking memory validation...")

    research_memory.clear()

    result = save_research_note.invoke(
        {
            "claim": "",
            "source": "Test Source",
            "topic": "structure-test",
        }
    )

    print(f"Empty claim result: {result}")

    assert "Error: claim cannot be empty." in result

    notes = get_research_notes.invoke(
        {
            "topic": "structure-test"
        }
    )

    print(f"Memory after invalid save: {notes}")

    assert "No previous research notes found" in notes

    print("Empty claim protection: OK")


def main():

    print("=" * 70)
    print("MARKET RESEARCH AGENT - STRUCTURE TEST")
    print("=" * 70)

    test_graph()
    test_tools()
    test_calculator()
    test_memory()
    test_empty_memory_validation()

    print("\n" + "=" * 70)
    print("ALL STRUCTURE CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()