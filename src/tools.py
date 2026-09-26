import json

from langchain.tools import tool
from tavily import TavilyClient

from .config import TAVILY_API_KEY
from .memory import ResearchMemory

tavily_client = TavilyClient(
    api_key=TAVILY_API_KEY
)


@tool
def web_search(query: str) -> str:
    """
    Search the web for current market, competitor, industry,
    pricing, customer, and business information.
    Use this tool whenever fresh external information is needed.
    """

    response = tavily_client.search(
        query=query,
        max_results=5,
        search_depth="advanced",
    )

    results = response.get("results", [])

    cleaned_results = []

    for result in results:
        cleaned_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
            }
        )

    return json.dumps(
        cleaned_results,
        ensure_ascii=False,
    )
#######################
@tool
def calculate_market_metrics(
    metric: str,
    initial_value: float,
    final_value: float,
    years: float = 1.0,
) -> str:
    """
    Calculate common market research metrics such as CAGR,
    percentage change, and absolute growth.
    """

    if initial_value <= 0:
        return "Error: initial_value must be greater than 0."

    if years <= 0:
        return "Error: years must be greater than 0."

    metric = metric.lower().strip()

    if metric == "cagr":
        cagr = ((final_value / initial_value) ** (1 / years) - 1) * 100
        return f"CAGR: {cagr:.2f}%"

    elif metric == "percentage_change":
        percentage_change = (
            (final_value - initial_value) / initial_value
        ) * 100
        return f"Percentage change: {percentage_change:.2f}%"

    elif metric == "absolute_growth":
        growth = final_value - initial_value
        return f"Absolute growth: {growth:.2f}"

    else:
        return (
            "Error: unsupported metric. "
            "Use 'cagr', 'percentage_change', or 'absolute_growth'."
        )
###########################33
research_memory = ResearchMemory()
@tool
def save_research_note(
    claim: str,
    source: str,
    topic: str,
    date: str = "",
) -> str:
    """
    Save an evidence-based research finding for later use.
    """

    result = research_memory.save_note(
        claim=claim,
        source=source,
        topic=topic,
        date=date or None,
    )

    return str(result)

##############################
@tool
def get_research_notes(topic: str) -> str:
    """
    Retrieve previously saved research findings for a topic.
    """

    notes = research_memory.get_notes(topic)

    if not notes:
        return "No previous research notes found for this topic."

    return str(notes)
tools = [
    web_search,
    calculate_market_metrics,
    save_research_note,
    get_research_notes,
]
#############################33
