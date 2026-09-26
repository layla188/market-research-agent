import json

from langchain.tools import tool
from tavily import TavilyClient

from .config import TAVILY_API_KEY


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


tools = [
    web_search
]