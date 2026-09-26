import json
from datetime import date

from .agent import llm, research_agent
from .state import ResearchState

from langgraph.types import interrupt

def planner_node(state: ResearchState):
    topic = state["topic"]

    prompt = f"""
You are a market research planner.

Current date:
{date.today().isoformat()}

Research objective:
{topic}

Create a focused research plan.

Break the objective into 4 to 6 research tasks.

The tasks should cover areas such as:
- market size and growth
- industry trends
- competitors
- customer needs
- pricing or business models when relevant
- risks, barriers, or opportunities

Do not perform the research yet.

Return ONLY a valid JSON array of strings.
"""

    response = llm.invoke(prompt)

    plan = json.loads(response.content)

    return {
        "research_plan": plan,
        "iteration": 0,
    }

###############################################
def researcher_node(state: ResearchState):

    topic = state["topic"]
    research_plan = state["research_plan"]

    plan_text = "\n".join(
        f"{i}. {task}"
        for i, task in enumerate(
            research_plan,
            start=1,
        )
    )

    research_request = f"""
Research objective:

{topic}

Research plan:

{plan_text}

Perform the research needed to address this plan.

Use web search whenever external or current information
is required.

Search multiple times if necessary.

For every important finding:
- explain the finding clearly
- include the source URL
- distinguish factual evidence from your interpretation

Do not write the final polished market report yet.

Return detailed research notes that another analyst
can use later.
"""

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

    return {
        "research_results": [
            final_message.content
        ]
    }

##############################################
def analyzer_node(state: ResearchState):

    topic = state["topic"]
    research_results = state["research_results"]

    research_text = "\n\n".join(
        research_results
    )

    prompt = f"""
You are a senior market research analyst.

Research objective:

{topic}

Collected research evidence:

{research_text}

Analyze the evidence.

Your job is NOT to search for new information.
Use only the evidence provided above.

Identify:

1. Key market trends
2. Competitive patterns
3. Customer or buyer needs
4. Market barriers and risks
5. Potential market gaps
6. Business opportunities
7. Important uncertainties or weak evidence

Clearly distinguish:

FACTS:
What is directly supported by the research evidence.

INTERPRETATIONS:
What can reasonably be inferred from those facts.

Do not invent statistics, companies, pricing,
or unsupported claims.

Return a structured analysis that can later
be used to write a market research report.
"""

    response = llm.invoke(prompt)

    return {
        "analysis": response.content
    }

##################################################3
def writer_node(state: ResearchState):

    topic = state["topic"]

    research_text = "\n\n".join(
        state["research_results"]
    )

    analysis = state["analysis"]

    current_draft = state["draft_report"]
    critique = state["critique"]

    is_revision = bool(
        current_draft and critique
    )


    # --------------------------------------------------
    # First draft
    # --------------------------------------------------

    if not is_revision:

        prompt = f"""
You are a professional market research writer.

Research objective:

{topic}


RESEARCH EVIDENCE:

{research_text}


MARKET ANALYSIS:

{analysis}


Write a professional draft market research report.

Structure:

# Executive Summary

# Market Overview

# Market Trends

# Competitive Landscape

# Customer Needs

# Risks and Barriers

# Market Opportunities

# Key Uncertainties

# Sources


Rules:

- Base factual claims on supplied evidence.
- Do not invent statistics.
- Do not invent companies.
- Distinguish facts from interpretations.
- Include source URLs when available.
- Explicitly mention weak or incomplete evidence.
"""

        response = llm.invoke(prompt)

        return {
            "draft_report": response.content
        }


    # --------------------------------------------------
    # Revision
    # --------------------------------------------------

    prompt = f"""
You are revising a market research report.

Research objective:

{topic}


ORIGINAL EVIDENCE:

{research_text}


ANALYSIS:

{analysis}


CURRENT REPORT:

{current_draft}


REVIEWER CRITIQUE:

{critique}


Revise the report to address the reviewer critique.

Rules:

- Preserve correct information.
- Remove unsupported claims.
- Remove irrelevant companies.
- Correct contradictions.
- Do not invent missing evidence.
- If evidence is insufficient, state that explicitly.
- Keep source URLs where possible.

Return the COMPLETE revised report.
"""

    response = llm.invoke(prompt)

    return {
        "draft_report": response.content,
        "iteration": state["iteration"] + 1,
        "critique": "",
    }

################################################33
def reviewer_node(state: ResearchState):

    topic = state["topic"]
    research_results = "\n\n".join(
        state["research_results"]
    )
    analysis = state["analysis"]
    draft_report = state["draft_report"]

    prompt = f"""
You are a strict market research reviewer.

Research objective:

{topic}


ORIGINAL RESEARCH EVIDENCE:

{research_results}


ANALYSIS:

{analysis}


DRAFT REPORT:

{draft_report}


Review the report for:

1. factual support
2. relevance to the research objective
3. unsupported claims
4. hallucinated statistics
5. irrelevant competitors or companies
6. contradictions
7. missing important findings
8. separation between facts and interpretations
9. source quality and source coverage
10. clarity and usefulness of the report


Decide one of:

PASS
- The report is sufficiently supported and useful.

REVISE
- The report contains important problems that should be fixed.


Return ONLY valid JSON in this format:

{{
    "status": "PASS or REVISE",
    "critique": "Clear explanation of what is correct or what must be fixed."
}}
"""

    response = llm.invoke(prompt)

    review = json.loads(response.content)

    return {
        "review_status": review["status"],
        "critique": review["critique"],
    }

#####################################
MAX_REVISIONS = 2


def review_router(state: ResearchState):

    if state["review_status"] == "PASS":
        return "approved"

    if state["iteration"] >= MAX_REVISIONS:
        return "max_revisions"

    return "revise"

##################################
def max_revisions_node(state: ResearchState):

    return {
        "critique": (
            "Maximum revision limit reached. "
            "The report still has unresolved review issues. "
            + state["critique"]
        )
    }
#######################################
def human_approval_node(state: ResearchState):

    decision = interrupt(
        {
            "message": "Please review the market research report.",
            "report": state["draft_report"],
            "options": [
                "approve",
                "reject",
            ],
        }
    )

    approved = (
        str(decision).lower() == "approve"
    )

    return {
        "approved": approved
    }