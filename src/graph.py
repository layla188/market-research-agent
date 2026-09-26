from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import InMemorySaver

from .state import ResearchState

from .nodes import (
    planner_node,
    researcher_node,
    analyzer_node,
    analysis_router,
    writer_node,
    reviewer_node,
    review_router,
    max_revisions_node,
    human_approval_node,
)


# ============================================================
# Build graph
# ============================================================

builder = StateGraph(ResearchState)


# ============================================================
# Nodes
# ============================================================

builder.add_node(
    "planner",
    planner_node,
)

builder.add_node(
    "researcher",
    researcher_node,
)

builder.add_node(
    "analyzer",
    analyzer_node,
)

builder.add_node(
    "writer",
    writer_node,
)

builder.add_node(
    "reviewer",
    reviewer_node,
)

builder.add_node(
    "human_approval",
    human_approval_node,
)

builder.add_node(
    "max_revisions",
    max_revisions_node,
)


# ============================================================
# Initial workflow
# ============================================================

builder.add_edge(
    START,
    "planner",
)

builder.add_edge(
    "planner",
    "researcher",
)

builder.add_edge(
    "researcher",
    "analyzer",
)


# ============================================================
# Analyzer → Researcher / Writer loop
# ============================================================

builder.add_conditional_edges(
    "analyzer",
    analysis_router,
    {
        "research_more": "researcher",
        "continue": "writer",
    },
)


# ============================================================
# Writer → Reviewer
# ============================================================

builder.add_edge(
    "writer",
    "reviewer",
)


# ============================================================
# Reviewer routing
# ============================================================

builder.add_conditional_edges(
    "reviewer",
    review_router,
    {
        "approved": "human_approval",
        "revise": "writer",
        "max_revisions": "max_revisions",
    },
)


# ============================================================
# End paths
# ============================================================

builder.add_edge(
    "max_revisions",
    "human_approval",
)

builder.add_edge(
    "human_approval",
    END,
)


# ============================================================
# Checkpointing
# ============================================================

checkpointer = InMemorySaver()

research_graph = builder.compile(
    checkpointer=checkpointer
)