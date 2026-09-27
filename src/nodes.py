import json
from datetime import date
from langgraph.types import interrupt

from .agent import research_agent
from .config import OPENROUTER_API_KEY, OPENROUTER_MODEL
from langchain_openai import ChatOpenAI

from .tools import research_memory
from .state import ResearchState


llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    use_responses_api=False,
)


MAX_RESEARCH_ITERATIONS = 2
MAX_REVISIONS = 2


# ============================================================
# Planner
# ============================================================

def planner_node(state: ResearchState):

    topic = state["topic"]

    prompt = f"""
You are planning a market research project.

Research topic:
{topic}

Create a focused research plan covering:

1. Market size and growth
2. Market trends
3. Competitors
4. Customer needs and behavior
5. Pricing and business models
6. Risks, barriers, and opportunities

Return ONLY a JSON array containing 4-6 research tasks.

Example:
[
    "Research current market size and growth",
    "Identify major competitors",
    "Investigate customer needs"
]
"""

    response = llm.invoke(prompt)

    try:
        research_plan = json.loads(response.content)

        if not isinstance(research_plan, list):
            research_plan = []

    except json.JSONDecodeError:
        research_plan = []

    return {
        "research_plan": research_plan,
        "research_iteration": 0,
        "research_gaps": [],
        "evidence_status": "",
        "iteration": 0,
    }


# ============================================================
# Researcher
# ============================================================

def researcher_node(state: ResearchState):
    topic = state["topic"]
    research_plan = state["research_plan"]

    # -----------------------------------------------------
    # Load previous research memory
    # -----------------------------------------------------
    try:
        previous_notes = research_memory.get_notes(topic)
    except Exception as e:
        previous_notes = []
        return_state_error = f"Memory retrieval failed: {str(e)}"
    else:
        return_state_error = None

    # -----------------------------------------------------
    # Format research plan
    # -----------------------------------------------------
    plan_text = "\n".join(
        f"{i}. {task}"
        for i, task in enumerate(
            research_plan,
            start=1,
        )
    )

    # -----------------------------------------------------
    # Format previous memory
    # -----------------------------------------------------
    if previous_notes:
        memory_text = json.dumps(
            previous_notes,
            ensure_ascii=False,
            indent=2,
        )
    else:
        memory_text = (
            "No previous research notes are available "
            "for this topic."
        )

    # -----------------------------------------------------
    # Research request
    # -----------------------------------------------------
    research_request = f"""
Research objective:
{topic}
Research plan:
{plan_text}
PREVIOUS RESEARCH MEMORY:
{memory_text}
Perform the research needed to address the research plan.
Use the previous research memory to understand what has
already been discovered.
Do NOT unnecessarily repeat research that is already
well-supported.
Instead, focus on:

missing information
weak evidence
conflicting information
areas that need verification
research tasks that are not yet covered
TOOL USAGE:
Use web_search whenever current or external information
is required.
Use calculate_market_metrics when numerical market
metrics such as CAGR or percentage growth need to be
calculated.
Use save_research_note to save important evidence-based
findings that should remain available for later research.
Use get_research_notes if you need to retrieve previous
research findings.
RESEARCH RULES:
For every important finding:

explain the finding clearly
include the source URL
distinguish factual evidence from interpretation
prefer multiple independent sources when possible
Do not invent:

statistics
company information
pricing
sources
calculations
unsupported claims
Do not write the final polished market report yet.
Return detailed research notes that another analyst
can use later.
"""
    # -----------------------------------------------------
    # Run research agent
    # -----------------------------------------------------
    try:
        result = research_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": research_request,
                    }
                ]
            }
        )

        final_message = result["messages"][-1]
        research_output = final_message.content

    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Researcher error: {str(e)}")

        return {
            "research_results": [],
            "memory_notes": previous_notes,
            "errors": errors,
        }

    # -----------------------------------------------------
    # Save useful research output into memory
    # -----------------------------------------------------
    try:
        research_memory.save_note(
            claim=research_output,
            source="Research Agent",
            topic=topic,
            date=date.today().isoformat(),
        )
    except Exception as e:
        errors = list(state.get("errors", []))
        errors.append(f"Memory save failed: {str(e)}")

        if return_state_error:
            errors.append(return_state_error)

        return {
            "research_results": [research_output],
            "memory_notes": previous_notes,
            "errors": errors,
        }

    # -----------------------------------------------------
    # Return updated state
    # -----------------------------------------------------
    errors = list(state.get("errors", []))

    if return_state_error:
        errors.append(return_state_error)

    updated_memory = research_memory.get_notes(topic)

    return {
        "research_results": [research_output],
        "memory_notes": updated_memory,
        "errors": errors,
    }


# ============================================================
# Analyzer
# ============================================================

def analyzer_node(state: ResearchState):

    topic = state["topic"]
    research_results = state.get("research_results", [])

    research_text = "\n\n--- RESEARCH PASS ---\n\n".join(
        research_results
    )

    prompt = f"""
You are a senior market research analyst.

Research topic:
{topic}

Collected research:

{research_text}

Analyze the evidence and determine whether the research
is sufficient to produce a reliable market research report.

You must:

1. Identify important market trends.
2. Identify competitive patterns.
3. Identify customer needs.
4. Identify risks and barriers.
5. Identify potential opportunities.
6. Identify important uncertainties.
7. Distinguish FACTS from INTERPRETATIONS.
8. Check whether important claims have supporting evidence.
9. Identify missing information that would materially
   improve the report.

IMPORTANT:

If important evidence is still missing, mark the research
as INSUFFICIENT.

If the available evidence is enough to produce a useful
report, mark it as SUFFICIENT.

Return ONLY valid JSON in exactly this structure:

{{
    "status": "SUFFICIENT",
    "research_gaps": [],
    "analysis": "Your detailed evidence-based analysis..."
}}

OR:

{{
    "status": "INSUFFICIENT",
    "research_gaps": [
        "Specific missing evidence",
        "Another missing evidence item"
    ],
    "analysis": "Your current analysis and explanation..."
}}

Do not invent missing information.
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    # --------------------------------------------------------
    # Remove markdown JSON fences if the model adds them
    # --------------------------------------------------------

    if content.startswith("```json"):
        content = content[7:]

    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # --------------------------------------------------------
    # Parse structured analyzer response
    # --------------------------------------------------------

    try:

        parsed = json.loads(content)

        evidence_status = parsed.get(
            "status",
            "SUFFICIENT",
        )

        research_gaps = parsed.get(
            "research_gaps",
            [],
        )

        analysis = parsed.get(
            "analysis",
            "",
        )

        if evidence_status not in {
            "SUFFICIENT",
            "INSUFFICIENT",
        }:
            evidence_status = "SUFFICIENT"

        if not isinstance(research_gaps, list):
            research_gaps = []

    except json.JSONDecodeError:

        # Safe fallback if the LLM does not return valid JSON.
        evidence_status = "SUFFICIENT"
        research_gaps = []
        analysis = response.content

    return {
        "evidence_status": evidence_status,
        "research_gaps": research_gaps,
        "analysis": analysis,
    }


# ============================================================
# Analyzer Router
# ============================================================

def analysis_router(state: ResearchState):

    evidence_status = state.get(
        "evidence_status",
        "SUFFICIENT",
    )

    research_iteration = state.get(
        "research_iteration",
        0,
    )

    if (
        evidence_status == "INSUFFICIENT"
        and research_iteration < MAX_RESEARCH_ITERATIONS
    ):
        return "research_more"

    return "continue"


# ============================================================
# Writer
# ============================================================

def writer_node(state: ResearchState):

    topic = state["topic"]

    research_results = state.get(
        "research_results",
        [],
    )

    analysis = state.get(
        "analysis",
        "",
    )

    critique = state.get(
        "critique",
        "",
    )

    current_draft = state.get(
        "draft_report",
        "",
    )

    research_text = "\n\n".join(
        research_results
    )

    # --------------------------------------------------------
    # First draft
    # --------------------------------------------------------

    if not current_draft:

        prompt = f"""
You are a professional market research report writer.

Topic:
{topic}

Research evidence:
{research_text}

Analysis:
{analysis}

Create a clear evidence-based market research report.

Use these sections:

1. Executive Summary
2. Market Overview
3. Market Trends
4. Competitive Landscape
5. Customer Needs
6. Risks and Barriers
7. Market Opportunities
8. Key Uncertainties
9. Sources

Rules:

- Do not invent statistics.
- Do not invent companies.
- Do not invent pricing.
- Distinguish facts from interpretations.
- Include source URLs.
- Clearly mention weak or incomplete evidence.
- Base conclusions only on the research provided.
"""

    # --------------------------------------------------------
    # Revision
    # --------------------------------------------------------

    else:

        prompt = f"""
You are revising a market research report.

Topic:
{topic}

Current report:
{current_draft}

Reviewer critique:
{critique}

Research evidence:
{research_text}

Analysis:
{analysis}

Revise the report to address the reviewer's critique.

Rules:

- Do not invent evidence.
- Do not invent statistics.
- Do not invent sources.
- Preserve supported findings.
- Improve clarity and evidence coverage.
- Clearly distinguish facts from interpretations.
- Keep the same professional report structure.
"""

    response = llm.invoke(prompt)

    return {
        "draft_report": response.content,
        "critique": "",
        "iteration": (
            state.get("iteration", 0) + 1
            if current_draft
            else state.get("iteration", 0)
        ),
    }


# ============================================================
# Reviewer
# ============================================================

def reviewer_node(state: ResearchState):

    topic = state["topic"]

    draft_report = state.get(
        "draft_report",
        "",
    )

    prompt = f"""
You are a strict reviewer of a market research report.

Topic:
{topic}

Report:
{draft_report}

Evaluate the report for:

1. Factual support
2. Relevance
3. Unsupported claims
4. Hallucinated statistics
5. Irrelevant competitors
6. Contradictions
7. Missing important findings
8. Facts vs interpretations
9. Source quality and coverage
10. Clarity and usefulness

Return ONLY valid JSON:

{{
    "status": "PASS",
    "critique": "..."
}}

OR:

{{
    "status": "REVISE",
    "critique": "Specific changes required..."
}}
"""

    response = llm.invoke(prompt)

    content = response.content.strip()

    if content.startswith("```json"):
        content = content[7:]

    elif content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    try:

        parsed = json.loads(content)

        status = parsed.get(
            "status",
            "REVISE",
        )

        critique = parsed.get(
            "critique",
            "",
        )

        if status not in {
            "PASS",
            "REVISE",
        }:
            status = "REVISE"

    except json.JSONDecodeError:

        status = "REVISE"
        critique = response.content

    return {
        "review_status": status,
        "critique": critique,
    }


# ============================================================
# Reviewer Router
# ============================================================

def review_router(state: ResearchState):

    if state["review_status"] == "PASS":
        return "approved"

    if state["iteration"] >= MAX_REVISIONS:
        return "max_revisions"

    return "revise"


# ============================================================
# Maximum Revision Node
# ============================================================

def max_revisions_node(state: ResearchState):

    return {
        "critique": (
            "Maximum report revision attempts reached. "
            "Proceeding to human approval with the current report."
        )
    }


# ============================================================
# Human Approval
# ============================================================

def human_approval_node(state: ResearchState):

    decision = interrupt(
        {
            "type": "report_approval",
            "report": state["draft_report"],
            "message": (
                "Review the generated market research report "
                "before finalizing it."
            ),
            "options": [
                "approve",
                "reject",
            ],
        }
    )

    return {
        "approved": decision == "approve"
    }