from src.tools import (
    calculate_market_metrics,
    save_research_note,
    get_research_notes,
)


print("=" * 70)
print("TESTING NEW PROJECT ADDITIONS")
print("=" * 70)


# --------------------------------------------------
# 1. Test metrics tool
# --------------------------------------------------

print("\n[1] Testing calculate_market_metrics...")

cagr_result = calculate_market_metrics.invoke({
    "metric": "cagr",
    "initial_value": 12,
    "final_value": 47.82,
    "years": 6,
})

print("CAGR result:")
print(cagr_result)


growth_result = calculate_market_metrics.invoke({
    "metric": "percentage_change",
    "initial_value": 12,
    "final_value": 47.82,
    "years": 6,
})

print("\nPercentage change result:")
print(growth_result)


# --------------------------------------------------
# 2. Test research memory - save
# --------------------------------------------------

print("\n[2] Testing save_research_note...")

save_result = save_research_note.invoke({
    "claim": "The AI customer support software market is growing rapidly.",
    "source": "MarketsandMarkets",
    "topic": "AI-powered customer support software market",
    "date": "2026-09-27",
})

print("Save result:")
print(save_result)


# --------------------------------------------------
# 3. Test research memory - retrieve
# --------------------------------------------------

print("\n[3] Testing get_research_notes...")

notes_result = get_research_notes.invoke({
    "topic": "AI-powered customer support software market",
})

print("Retrieved notes:")
print(notes_result)


# --------------------------------------------------
# 4. Test guardrail - invalid calculator input
# --------------------------------------------------



print("\n" + "=" * 70)
print("ADDITION TEST FINISHED")
print("=" * 70)