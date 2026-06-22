"""
graph.py
--------
Wires the six agents from agents.py together into a LangGraph StateGraph.

Why this file exists on its own:
    This file answers ONE question only: "in what order do agents run, and
    under what conditions do we branch or loop?" It contains zero business
    logic of its own -- it only imports node functions from agents.py and
    connects them. This separation means you can redraw/rewire the workflow
    (e.g. add a new agent, change the loop condition) without touching any
    agent's internal logic, and vice versa.

This builds exactly the workflow from the problem statement:

    Business Question
           |
    Intent Analyzer
           |
    SQL Generator  <------------------------+
           |                                |
    Data Retrieval                          |
           |                                |
    Analysis Agent                          |
           |                                |
    Need More Data? --- Yes ----------------+
           |
           No
           |
    Visualization
           |
    Final Report
"""

from langgraph.graph import StateGraph, END

from state import AgentState
from agents import (
    intent_analyzer_agent,
    sql_generator_agent,
    data_retrieval_agent,
    analysis_agent,
    visualization_agent,
    report_writer_agent,
    route_after_analysis,
)


def build_graph():
    """
    Constructs and compiles the LangGraph workflow.

    Returns a compiled graph object that supports two execution modes:
      - app.invoke(state)  -> runs the full graph and returns only the
                              final state (used by main.py, the CLI entry point)
      - app.stream(state)  -> yields the state update after EVERY node
                              finishes, one node at a time (used by app.py,
                              the Streamlit UI, to show live per-agent progress)

    Both modes run the exact same graph -- streaming does not change the
    workflow logic, it only changes how much visibility into intermediate
    steps the caller gets.
    """
    workflow = StateGraph(AgentState)

    # ---- Register every agent as a node ----
    # node name (string)        node function (callable)
    workflow.add_node("intent_analyzer", intent_analyzer_agent)
    workflow.add_node("sql_generator", sql_generator_agent)
    workflow.add_node("data_retrieval", data_retrieval_agent)
    workflow.add_node("analysis", analysis_agent)
    workflow.add_node("visualization", visualization_agent)
    workflow.add_node("report_writer", report_writer_agent)

    # ---- Entry point: where execution starts ----
    workflow.set_entry_point("intent_analyzer")

    # ---- Linear edges (straightforward, unconditional handoffs) ----
    workflow.add_edge("intent_analyzer", "sql_generator")
    workflow.add_edge("sql_generator", "data_retrieval")
    workflow.add_edge("data_retrieval", "analysis")

    # ---- Conditional edge: this IS the "Need More Data?" diamond ----
    # After `analysis` runs, `route_after_analysis` inspects state and
    # returns either "sql_generator" (loop back) or "visualization" (move
    # forward). LangGraph uses that returned string to pick the next node.
    workflow.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {
            "sql_generator": "sql_generator",  # loop back for a follow-up query
            "visualization": "visualization",  # enough data, proceed
        },
    )

    # ---- Remaining linear edges ----
    workflow.add_edge("visualization", "report_writer")
    workflow.add_edge("report_writer", END)

    return workflow.compile()
