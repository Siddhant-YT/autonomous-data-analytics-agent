"""
state.py
--------
Defines AgentState: the single shared data structure that flows through
every node in the LangGraph.

Why this file exists on its own:
    In LangGraph, the "state" is the contract between all agents. Every
    node receives the current state, does its job, and returns a dict of
    fields to update. Keeping this definition in its own file means:
      1. It documents, in one place, every piece of data the agents
         exchange -- effectively the "API" of the whole system.
      2. agents.py and graph.py both import from here, so there is a
         single source of truth for what the state looks like.

Think of AgentState as a clipboard that gets passed from agent to agent.
Each agent reads what it needs off the clipboard and writes its own
findings back onto it before passing it along.
"""

from typing import TypedDict, List, Optional


class AgentState(TypedDict, total=False):
    # ----- Input -----
    question: str
    # The original business question, e.g. "Why did sales drop in Q1?"

    # ----- Set by Intent Analyzer Agent -----
    intent: str
    # A short structured summary of what the user actually wants,
    # e.g. "Investigate revenue decline, focus on region/category/time"

    focus_entities: List[str]
    # Key entities the LLM extracted from the question, e.g.
    # ["sales", "drop", "region", "time period"]

    # ----- Set by SQL Generator Agent -----
    sql_query: str
    # The SQL query generated for the current iteration

    sql_history: List[str]
    # Every SQL query generated across all loop iterations (useful for
    # the final report and for debugging / demoing to your manager)

    # ----- Set by Data Retrieval Agent -----
    raw_data: List[dict]
    # Rows returned from SQLite, as a list of dicts (easy to inspect/print)

    row_count: int
    # Number of rows retrieved -- used by the Analysis Agent to judge
    # whether the data sample is sufficient

    # ----- Set by Analysis Agent -----
    insights: List[str]
    # Bullet-point findings extracted from the data, accumulated across
    # loop iterations

    needs_more_data: bool
    # The routing decision: should we loop back to SQL Generator for a
    # follow-up, more targeted query?

    follow_up_reason: Optional[str]
    # WHY the Analysis Agent thinks it needs more data -- this becomes the
    # instruction passed back into the SQL Generator on the next loop

    loop_count: int
    # How many times we've looped back. Used to enforce MAX_REFINEMENT_LOOPS
    # so the graph can never run forever.

    # ----- Set by Visualization Agent -----
    chart_paths: List[str]
    # File paths of charts (PNG) generated for the report

    chart_descriptions: List[str]
    # One-line description of what each chart shows

    # ----- Set by Report Writer Agent -----
    final_report: str
    # The polished, human-readable Markdown report -- the final output
    # of the entire pipeline
