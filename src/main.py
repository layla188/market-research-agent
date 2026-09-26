'''
from src.graph import research_graph


print("Graph compiled successfully.")

print("\nGraph structure:")
print(research_graph.get_graph().draw_ascii())
'''
from src.graph import research_graph


config = {
    "configurable": {
        "thread_id": "test-market-research-1"
    }
}


initial_state = {
    "topic": "AI-powered customer support software market",
    "research_plan": [],
    "research_results": [],
    "research_gaps": [],
    "research_iteration": 0,
    "evidence_status": "",
    "analysis": "",
    "draft_report": "",
    "critique": "",
    "review_status": "",
    "iteration": 0,
    "approved": False,
}


result = research_graph.invoke(
    initial_state,
    config=config,
)


print("\n" + "=" * 70)
print("RESEARCH FINISHED")
print("=" * 70)

print("\nResearch iterations:")
print(result.get("research_iteration"))

print("\nEvidence status:")
print(result.get("evidence_status"))

print("\nResearch gaps:")
print(result.get("research_gaps"))

print("\nReview status:")
print(result.get("review_status"))

print("\nDraft report:")
print(result.get("draft_report"))