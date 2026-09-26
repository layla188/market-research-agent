from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from .state import ResearchState

from .nodes import (
    planner_node,
    researcher_node,
    analyzer_node,
    writer_node,
    reviewer_node,
    max_revisions_node,
    human_approval_node,
    review_router,
)


builder = StateGraph(
    ResearchState
)


# -------------------------------------------
# Nodes
# -------------------------------------------

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


# -------------------------------------------
# Normal edges
# -------------------------------------------

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

builder.add_edge(
    "analyzer",
    "writer",
)

builder.add_edge(
    "writer",
    "reviewer",
)


# -------------------------------------------
# Conditional edge
# -------------------------------------------

builder.add_conditional_edges(
    "reviewer",
    review_router,
    {
        "approved": "human_approval",
        "revise": "writer",
        "max_revisions": "max_revisions",
    },
)


builder.add_edge(
    "human_approval",
    END,
)

builder.add_edge(
    "max_revisions",
    "human_approval",
)


# -------------------------------------------
# Memory / checkpointing
# -------------------------------------------

checkpointer = InMemorySaver()


research_graph = builder.compile(
    checkpointer=checkpointer
)