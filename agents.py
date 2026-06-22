# """
# agents.py
# ---------
# Defines all six agents as LangGraph "node" functions.

# Why this file exists on its own:
#     LangGraph nodes are just plain Python functions with the signature
#         def node(state: AgentState) -> dict
#     They read fields off `state` and return a dict of fields to update.
#     Keeping ALL agent logic here (separate from graph.py, which only
#     wires nodes together) means you can unit-test or reason about each
#     agent in total isolation from the orchestration logic.

# Each agent below follows the same internal pattern:
#     1. Pull what it needs from `state`.
#     2. Build a prompt and call the shared `llm`.
#     3. Parse the LLM's response into something structured.
#     4. Return a dict of state updates (LangGraph merges this into state).

# The six agents, in pipeline order:
#     1. intent_analyzer_agent   -- understands the business question
#     2. sql_generator_agent     -- writes a SQL query to answer it
#     3. data_retrieval_agent    -- executes the query against SQLite
#     4. analysis_agent          -- finds trends/insights, decides if more
#                                   data is needed (drives the loop)
#     5. visualization_agent     -- creates charts from the retrieved data
#     6. report_writer_agent     -- writes the final Markdown report
# """

# import json
# import sqlite3
# import os
# import re
# from datetime import datetime

# import matplotlib
# matplotlib.use("Agg")  # non-interactive backend, safe for scripts/servers
# import matplotlib.pyplot as plt
# import pandas as pd

# from langchain_core.messages import SystemMessage, HumanMessage

# from config import llm, DB_PATH, OUTPUT_DIR, MAX_REFINEMENT_LOOPS
# from database import get_schema_description
# from state import AgentState


# # ---------------------------------------------------------------------------
# # Small shared helper: ask the LLM for JSON and parse it safely.
# # LLMs occasionally wrap JSON in markdown fences or add stray text, so we
# # defensively strip that before parsing. Centralizing this avoids repeating
# # the same try/except in every agent.
# # ---------------------------------------------------------------------------
# def _call_llm_for_json(system_prompt: str, user_prompt: str) -> dict:
#     response = llm.invoke(
#         [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     )
#     text = response.content.strip()

#     # Strip ```json ... ``` or ``` ... ``` fences if present
#     text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()

#     try:
#         return json.loads(text)
#     except json.JSONDecodeError:
#         # Last resort: try to find the first {...} block in the text
#         match = re.search(r"\{.*\}", text, flags=re.DOTALL)
#         if match:
#             return json.loads(match.group(0))
#         raise ValueError(f"Could not parse JSON from LLM response:\n{text}")


# # ===========================================================================
# # AGENT 1: Intent Analyzer
# # ===========================================================================
# def intent_analyzer_agent(state: AgentState) -> dict:
#     """
#     Reads the raw business question and turns it into a structured intent:
#     what is the user really asking, and which entities (region, time,
#     category, metric) matter for answering it.

#     This matters because a vague question like "Why did sales drop?" is not
#     directly translatable to SQL. We need an explicit interpretation step
#     first, exactly like a real analyst would clarify the ask before
#     touching a database.
#     """
#     question = state["question"]

#     system_prompt = (
#         "You are a senior business analyst. Your job is to interpret a "
#         "business question and identify what data investigation is needed. "
#         "Respond ONLY with a valid JSON object, no markdown, no extra text. "
#         'Format: {"intent": "<one sentence summary of the analytical goal>", '
#         '"focus_entities": ["<entity1>", "<entity2>", ...]}'
#     )
#     user_prompt = (
#         f"Business question: \"{question}\"\n\n"
#         "Identify the analytical intent and the key entities relevant to "
#         "investigating it (e.g. metric of interest, time period, region, "
#         "product category)."
#     )

#     result = _call_llm_for_json(system_prompt, user_prompt)

#     print(f"[Intent Analyzer] intent: {result.get('intent')}")
#     print(f"[Intent Analyzer] focus_entities: {result.get('focus_entities')}")

#     return {
#         "intent": result.get("intent", ""),
#         "focus_entities": result.get("focus_entities", []),
#         "loop_count": 0,
#         "sql_history": [],
#         "insights": [],
#     }


# # ===========================================================================
# # AGENT 2: SQL Generator
# # ===========================================================================
# def sql_generator_agent(state: AgentState) -> dict:
#     """
#     Translates the analytical intent into a concrete SQL query against the
#     `sales` table.

#     On the FIRST pass, this writes a broad exploratory query.
#     On LOOP-BACK passes (when the Analysis Agent asked for more data), this
#     writes a more targeted follow-up query using `follow_up_reason`.
#     """
#     schema = get_schema_description(DB_PATH)
#     intent = state["intent"]
#     follow_up_reason = state.get("follow_up_reason")

#     system_prompt = (
#         "You are an expert SQLite query writer. Given a database schema and "
#         "an analytical goal, write ONE syntactically correct SQLite query "
#         "that helps answer it. Prefer including GROUP BY / aggregations "
#         "(SUM, AVG, COUNT) over raw row dumps so the result is analysis-ready. "
#         "Limit results to at most 200 rows. "
#         "Respond ONLY with a valid JSON object, no markdown, no extra text. "
#         'Format: {"sql": "<the SQL query>", "explanation": "<one sentence on what it retrieves>"}'
#     )

#     if follow_up_reason:
#         user_prompt = (
#             f"Schema:\n{schema}\n\n"
#             f"Original analytical goal: {intent}\n\n"
#             f"A previous analysis pass found this needs deeper investigation: "
#             f"\"{follow_up_reason}\"\n\n"
#             "Write a NEW, more targeted SQL query that drills into this "
#             "specific follow-up. Avoid repeating the exact same query as before."
#         )
#     else:
#         user_prompt = (
#             f"Schema:\n{schema}\n\n"
#             f"Analytical goal: {intent}\n\n"
#             "Write an exploratory SQL query to begin investigating this question."
#         )

#     result = _call_llm_for_json(system_prompt, user_prompt)
#     sql = result.get("sql", "").strip()

#     print(f"[SQL Generator] explanation: {result.get('explanation')}")
#     print(f"[SQL Generator] sql: {sql}")

#     history = state.get("sql_history", [])
#     history.append(sql)

#     return {"sql_query": sql, "sql_history": history}


# # ===========================================================================
# # AGENT 3: Data Retrieval
# # ===========================================================================
# def data_retrieval_agent(state: AgentState) -> dict:
#     """
#     Executes the SQL query generated by the previous agent against the
#     SQLite database and returns the rows as a list of dicts.

#     This agent does NOT call the LLM at all -- it is a pure execution step.
#     This is an important design point to highlight: not every node in a
#     LangGraph workflow needs to be "AI-powered". Deterministic steps
#     (like running SQL) belong in plain code, reserving LLM calls for steps
#     that genuinely require language understanding or generation.
#     """
#     sql = state["sql_query"]

#     try:
#         conn = sqlite3.connect(DB_PATH)
#         conn.row_factory = sqlite3.Row
#         cur = conn.cursor()
#         cur.execute(sql)
#         rows = [dict(row) for row in cur.fetchall()]
#         conn.close()
#         error = None
#     except sqlite3.Error as e:
#         rows = []
#         error = str(e)

#     print(f"[Data Retrieval] rows fetched: {len(rows)}" + (f" | ERROR: {error}" if error else ""))

#     return {
#         "raw_data": rows,
#         "row_count": len(rows),
#     }


# # ===========================================================================
# # AGENT 4: Analysis Agent  (drives the conditional loop)
# # ===========================================================================
# def analysis_agent(state: AgentState) -> dict:
#     """
#     Looks at the retrieved data and extracts insights/trends. Crucially,
#     this agent also DECIDES whether the data gathered so far is sufficient
#     to answer the original question, or whether the graph should loop back
#     to the SQL Generator for a more targeted follow-up query.

#     This is the agent that implements the "Need More Data?" diamond in the
#     workflow diagram. The decision (`needs_more_data`) is read by graph.py's
#     conditional edge to route execution.
#     """
#     intent = state["intent"]
#     data = state.get("raw_data", [])
#     loop_count = state.get("loop_count", 0)

#     # Hard safety stop regardless of what the LLM thinks -- prevents
#     # infinite loops if the model keeps asking for more data.
#     force_stop = loop_count >= MAX_REFINEMENT_LOOPS

#     # Sample the data for the prompt (avoid blowing up token usage on
#     # large result sets -- 30 rows is plenty for trend-spotting).
#     sample = data[:30]

#     system_prompt = (
#         "You are a data analyst. Given a business question and a sample of "
#         "query results, extract 2-4 concrete, specific insights (mention "
#         "actual numbers/regions/categories/dates where possible). Then decide "
#         "if the data is SUFFICIENT to confidently answer the business question, "
#         "or if a follow-up query digging into a specific dimension "
#         "(e.g. by region, by category, by time period) is needed. "
#         "Respond ONLY with a valid JSON object, no markdown, no extra text. "
#         'Format: {"insights": ["<insight1>", "<insight2>", ...], '
#         '"needs_more_data": <true|false>, '
#         '"follow_up_reason": "<specific reason/direction for follow-up query, or null>"}'
#     )
#     user_prompt = (
#         f"Business question / goal: {intent}\n\n"
#         f"Query result sample ({len(data)} total rows, showing up to 30):\n"
#         f"{json.dumps(sample, indent=2, default=str)}\n\n"
#         "Analyze this data."
#     )

#     if not data:
#         # No data returned at all -- definitely insufficient, but don't
#         # loop forever; let the safety stop catch it if it keeps happening.
#         result = {
#             "insights": ["The query returned no rows; the filters may be too narrow or the SQL may need adjustment."],
#             "needs_more_data": True,
#             "follow_up_reason": "Previous query returned zero rows; broaden or correct the filters.",
#         }
#     else:
#         result = _call_llm_for_json(system_prompt, user_prompt)

#     needs_more_data = bool(result.get("needs_more_data", False)) and not force_stop

#     if force_stop and result.get("needs_more_data"):
#         print(f"[Analysis Agent] Reached MAX_REFINEMENT_LOOPS ({MAX_REFINEMENT_LOOPS}) -- forcing stop.")

#     print(f"[Analysis Agent] insights: {result.get('insights')}")
#     print(f"[Analysis Agent] needs_more_data: {needs_more_data} (loop_count={loop_count})")

#     all_insights = state.get("insights", []) + result.get("insights", [])

#     return {
#         "insights": all_insights,
#         "needs_more_data": needs_more_data,
#         "follow_up_reason": result.get("follow_up_reason") if needs_more_data else None,
#         "loop_count": loop_count + 1,
#     }


# # ===========================================================================
# # Conditional routing function (used by graph.py, not itself a node)
# # ===========================================================================
# def route_after_analysis(state: AgentState) -> str:
#     """
#     Reads `needs_more_data` from state and returns the name of the next
#     node to visit. This function is registered with LangGraph's
#     `add_conditional_edges` and implements the "Need More Data? Yes/No"
#     diamond from the workflow diagram.
#     """
#     if state.get("needs_more_data"):
#         return "sql_generator"
#     return "visualization"


# # ===========================================================================
# # AGENT 5: Visualization Agent
# # ===========================================================================
# def visualization_agent(state: AgentState) -> dict:
#     """
#     Builds chart(s) from the most recent retrieved data using pandas +
#     matplotlib, and saves them as PNG files. This agent uses the LLM only
#     to decide WHICH columns are most meaningful to plot and HOW (chart
#     type, axis labels) -- the actual chart rendering is deterministic code.
#     """
#     data = state.get("raw_data", [])
#     intent = state["intent"]

#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     if not data:
#         return {"chart_paths": [], "chart_descriptions": ["No data available to visualize."]}

#     df = pd.DataFrame(data)
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     categorical_cols = [c for c in df.columns if c not in numeric_cols]

#     system_prompt = (
#         "You are a data visualization expert. Given the available columns "
#         "in a dataframe and the analytical goal, choose ONE good chart to "
#         "visualize the data. "
#         "Respond ONLY with a valid JSON object, no markdown, no extra text. "
#         'Format: {"chart_type": "<bar|line>", "x_column": "<column name>", '
#         '"y_column": "<column name>", "title": "<chart title>"}'
#     )
#     user_prompt = (
#         f"Analytical goal: {intent}\n"
#         f"Categorical/text columns available: {categorical_cols}\n"
#         f"Numeric columns available: {numeric_cols}\n"
#         f"Row count: {len(df)}\n\n"
#         "Pick the most informative column pair and chart type for this goal."
#     )

#     try:
#         choice = _call_llm_for_json(system_prompt, user_prompt)
#         x_col = choice.get("x_column")
#         y_col = choice.get("y_column")
#         chart_type = choice.get("chart_type", "bar")
#         title = choice.get("title", "Sales Analysis")

#         if x_col not in df.columns or y_col not in df.columns:
#             raise ValueError("LLM chose columns not present in the dataframe.")

#         # Aggregate if x_column has repeated values (common after GROUP BY
#         # queries that still have duplicate labels across loop iterations)
#         plot_df = df.groupby(x_col, as_index=False)[y_col].sum()
#         plot_df = plot_df.sort_values(by=y_col, ascending=False).head(15)

#         fig, ax = plt.subplots(figsize=(9, 5))
#         if chart_type == "line":
#             ax.plot(plot_df[x_col], plot_df[y_col], marker="o")
#         else:
#             ax.bar(plot_df[x_col], plot_df[y_col], color="#4C72B0")

#         ax.set_title(title)
#         ax.set_xlabel(x_col)
#         ax.set_ylabel(y_col)
#         plt.xticks(rotation=45, ha="right")
#         plt.tight_layout()

#         loop_count = state.get("loop_count", 0)
#         # chart_path = os.path.join(OUTPUT_DIR, f"chart_{loop_count}.png")
#         timestamp = datetime.now().strftime("%d-%b-%Y_%Hh%Mm")
#         chart_path = os.path.join(OUTPUT_DIR, f"chart_{timestamp}_{loop_count}.png")
#         plt.savefig(chart_path)
#         plt.close(fig)

#         description = f"{title} ({chart_type} chart of {y_col} by {x_col})"
#         print(f"[Visualization] saved chart: {chart_path} | {description}")

#         existing_paths = state.get("chart_paths", [])
#         existing_descs = state.get("chart_descriptions", [])

#         return {
#             "chart_paths": existing_paths + [chart_path],
#             "chart_descriptions": existing_descs + [description],
#         }

#     except Exception as e:
#         print(f"[Visualization] skipped chart due to: {e}")
#         return {
#             "chart_paths": state.get("chart_paths", []),
#             "chart_descriptions": state.get("chart_descriptions", []) + [f"Chart generation skipped: {e}"],
#         }


# # ===========================================================================
# # AGENT 6: Report Writer
# # ===========================================================================
# def report_writer_agent(state: AgentState) -> dict:
#     """
#     Synthesizes everything gathered by the previous agents -- the original
#     question, intent, all accumulated insights, SQL queries used, and chart
#     descriptions -- into one polished, decision-ready Markdown report.

#     This is the final node in the graph; its output (`final_report`) is
#     what gets shown to the business user.
#     """
#     question = state["question"]
#     intent = state["intent"]
#     insights = state.get("insights", [])
#     sql_history = state.get("sql_history", [])
#     chart_descriptions = state.get("chart_descriptions", [])

#     system_prompt = (
#         "You are a senior business analyst writing a final report for a "
#         "business stakeholder. Write in clear, confident, plain language. "
#         "Use the provided insights to construct a coherent narrative with "
#         "a clear conclusion and 1-2 actionable recommendations. "
#         "Output well-formatted Markdown with headings. Do not use emojis."
#     )
#     user_prompt = (
#         f"Original business question: \"{question}\"\n\n"
#         f"Analytical focus: {intent}\n\n"
#         f"Accumulated insights from data analysis:\n"
#         + "\n".join(f"- {i}" for i in insights)
#         + f"\n\nCharts produced: {', '.join(chart_descriptions) if chart_descriptions else 'none'}\n\n"
#         "Write the final report. Structure it with these sections: "
#         "Summary, Key Findings, Recommendation."
#     )

#     response = llm.invoke(
#         [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
#     )
#     report_body = response.content.strip()

#     # Append a technical appendix so your manager can see the actual SQL
#     # the system generated -- great for demonstrating transparency/auditability.
#     appendix_lines = ["", "---", "", "## Appendix: Queries Executed"]
#     for i, sql in enumerate(sql_history, start=1):
#         appendix_lines.append(f"\n**Query {i}:**\n```sql\n{sql}\n```")

#     final_report = report_body + "\n".join(appendix_lines)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     # report_path = os.path.join(OUTPUT_DIR, "final_report.md")
#     timestamp = datetime.now().strftime("%d-%b-%Y_%Hh%Mm")
#     report_path = os.path.join(OUTPUT_DIR, f"final_report_{timestamp}.md")
#     with open(report_path, "w") as f:
#         f.write(final_report)

#     print(f"[Report Writer] report saved to: {report_path}")

#     return {"final_report": final_report}

























"""
agents.py
---------
Defines all six agents as LangGraph "node" functions.

Why this file exists on its own:
    LangGraph nodes are just plain Python functions with the signature
        def node(state: AgentState) -> dict
    They read fields off `state` and return a dict of fields to update.
    Keeping ALL agent logic here (separate from graph.py, which only
    wires nodes together) means you can unit-test or reason about each
    agent in total isolation from the orchestration logic.

Each agent below follows the same internal pattern:
    1. Pull what it needs from `state`.
    2. Build a prompt and call the shared `llm`.
    3. Parse the LLM's response into something structured.
    4. Return a dict of state updates (LangGraph merges this into state).

The six agents, in pipeline order:
    1. intent_analyzer_agent   -- understands the business question
    2. sql_generator_agent     -- writes a SQL query to answer it
    3. data_retrieval_agent    -- executes the query against SQLite
    4. analysis_agent          -- finds trends/insights, decides if more
                                  data is needed (drives the loop)
    5. visualization_agent     -- creates charts from the retrieved data
    6. report_writer_agent     -- writes the final Markdown report
"""

import json
import sqlite3
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts/servers
import matplotlib.pyplot as plt
import pandas as pd

from langchain_core.messages import SystemMessage, HumanMessage

from config import llm, DB_PATH, OUTPUT_DIR, MAX_REFINEMENT_LOOPS
from database import get_schema_description
from state import AgentState


# ---------------------------------------------------------------------------
# Small shared helper: ask the LLM for JSON and parse it safely.
# LLMs occasionally wrap JSON in markdown fences or add stray text, so we
# defensively strip that before parsing. Centralizing this avoids repeating
# the same try/except in every agent.
# ---------------------------------------------------------------------------
def _call_llm_for_json(system_prompt: str, user_prompt: str) -> dict:
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    text = response.content.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: try to find the first {...} block in the text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse JSON from LLM response:\n{text}")


# ===========================================================================
# AGENT 1: Intent Analyzer
# ===========================================================================
def intent_analyzer_agent(state: AgentState) -> dict:
    """
    Reads the raw business question and turns it into a structured intent:
    what is the user really asking, and which entities (region, time,
    category, metric) matter for answering it.

    This matters because a vague question like "Why did sales drop?" is not
    directly translatable to SQL. We need an explicit interpretation step
    first, exactly like a real analyst would clarify the ask before
    touching a database.
    """
    question = state["question"]

    system_prompt = (
        "You are a senior business analyst. Your job is to interpret a "
        "business question and identify what data investigation is needed. "
        "Respond ONLY with a valid JSON object, no markdown, no extra text. "
        'Format: {"intent": "<one sentence summary of the analytical goal>", '
        '"focus_entities": ["<entity1>", "<entity2>", ...]}'
    )
    user_prompt = (
        f"Business question: \"{question}\"\n\n"
        "Identify the analytical intent and the key entities relevant to "
        "investigating it (e.g. metric of interest, time period, region, "
        "product category)."
    )

    result = _call_llm_for_json(system_prompt, user_prompt)

    print(f"[Intent Analyzer] intent: {result.get('intent')}")
    print(f"[Intent Analyzer] focus_entities: {result.get('focus_entities')}")

    return {
        "intent": result.get("intent", ""),
        "focus_entities": result.get("focus_entities", []),
        "loop_count": 0,
        "sql_history": [],
        "insights": [],
    }


# ===========================================================================
# AGENT 2: SQL Generator
# ===========================================================================
def sql_generator_agent(state: AgentState) -> dict:
    """
    Translates the analytical intent into a concrete SQL query against the
    `sales` table.

    On the FIRST pass, this writes a broad exploratory query.
    On LOOP-BACK passes (when the Analysis Agent asked for more data), this
    writes a more targeted follow-up query using `follow_up_reason`.
    """
    schema = get_schema_description(DB_PATH)
    intent = state["intent"]
    follow_up_reason = state.get("follow_up_reason")

    system_prompt = (
        "You are an expert SQLite query writer. Given a database schema and "
        "an analytical goal, write ONE syntactically correct SQLite query "
        "that helps answer it. Prefer including GROUP BY / aggregations "
        "(SUM, AVG, COUNT) over raw row dumps so the result is analysis-ready. "
        "Limit results to at most 200 rows. "
        "Respond ONLY with a valid JSON object, no markdown, no extra text. "
        'Format: {"sql": "<the SQL query>", "explanation": "<one sentence on what it retrieves>"}'
    )

    if follow_up_reason:
        user_prompt = (
            f"Schema:\n{schema}\n\n"
            f"Original analytical goal: {intent}\n\n"
            f"A previous analysis pass found this needs deeper investigation: "
            f"\"{follow_up_reason}\"\n\n"
            "Write a NEW, more targeted SQL query that drills into this "
            "specific follow-up. Avoid repeating the exact same query as before."
        )
    else:
        user_prompt = (
            f"Schema:\n{schema}\n\n"
            f"Analytical goal: {intent}\n\n"
            "Write an exploratory SQL query to begin investigating this question."
        )

    result = _call_llm_for_json(system_prompt, user_prompt)
    sql = result.get("sql", "").strip()

    print(f"[SQL Generator] explanation: {result.get('explanation')}")
    print(f"[SQL Generator] sql: {sql}")

    history = state.get("sql_history", [])
    history.append(sql)

    return {"sql_query": sql, "sql_history": history}


# ===========================================================================
# AGENT 3: Data Retrieval
# ===========================================================================
def data_retrieval_agent(state: AgentState) -> dict:
    """
    Executes the SQL query generated by the previous agent against the
    SQLite database and returns the rows as a list of dicts.

    This agent does NOT call the LLM at all -- it is a pure execution step.
    This is an important design point to highlight: not every node in a
    LangGraph workflow needs to be "AI-powered". Deterministic steps
    (like running SQL) belong in plain code, reserving LLM calls for steps
    that genuinely require language understanding or generation.
    """
    sql = state["sql_query"]

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql)
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        error = None
    except sqlite3.Error as e:
        rows = []
        error = str(e)

    print(f"[Data Retrieval] rows fetched: {len(rows)}" + (f" | ERROR: {error}" if error else ""))

    return {
        "raw_data": rows,
        "row_count": len(rows),
    }


# ===========================================================================
# AGENT 4: Analysis Agent  (drives the conditional loop)
# ===========================================================================
def analysis_agent(state: AgentState) -> dict:
    """
    Looks at the retrieved data and extracts insights/trends. Crucially,
    this agent also DECIDES whether the data gathered so far is sufficient
    to answer the original question, or whether the graph should loop back
    to the SQL Generator for a more targeted follow-up query.

    This is the agent that implements the "Need More Data?" diamond in the
    workflow diagram. The decision (`needs_more_data`) is read by graph.py's
    conditional edge to route execution.
    """
    intent = state["intent"]
    data = state.get("raw_data", [])
    loop_count = state.get("loop_count", 0)

    # Hard safety stop regardless of what the LLM thinks -- prevents
    # infinite loops if the model keeps asking for more data.
    force_stop = loop_count >= MAX_REFINEMENT_LOOPS

    # Sample the data for the prompt (avoid blowing up token usage on
    # large result sets -- 30 rows is plenty for trend-spotting).
    sample = data[:30]

    system_prompt = (
        "You are a data analyst. Given a business question and a sample of "
        "query results, extract 2-4 concrete, specific insights (mention "
        "actual numbers/regions/categories/dates where possible). Then decide "
        "if the data is SUFFICIENT to confidently answer the business question, "
        "or if a follow-up query digging into a specific dimension "
        "(e.g. by region, by category, by time period) is needed. "
        "Respond ONLY with a valid JSON object, no markdown, no extra text. "
        'Format: {"insights": ["<insight1>", "<insight2>", ...], '
        '"needs_more_data": <true|false>, '
        '"follow_up_reason": "<specific reason/direction for follow-up query, or null>"}'
    )
    user_prompt = (
        f"Business question / goal: {intent}\n\n"
        f"Query result sample ({len(data)} total rows, showing up to 30):\n"
        f"{json.dumps(sample, indent=2, default=str)}\n\n"
        "Analyze this data."
    )

    if not data:
        # No data returned at all -- definitely insufficient, but don't
        # loop forever; let the safety stop catch it if it keeps happening.
        result = {
            "insights": ["The query returned no rows; the filters may be too narrow or the SQL may need adjustment."],
            "needs_more_data": True,
            "follow_up_reason": "Previous query returned zero rows; broaden or correct the filters.",
        }
    else:
        result = _call_llm_for_json(system_prompt, user_prompt)

    needs_more_data = bool(result.get("needs_more_data", False)) and not force_stop

    if force_stop and result.get("needs_more_data"):
        print(f"[Analysis Agent] Reached MAX_REFINEMENT_LOOPS ({MAX_REFINEMENT_LOOPS}) -- forcing stop.")

    print(f"[Analysis Agent] insights: {result.get('insights')}")
    print(f"[Analysis Agent] needs_more_data: {needs_more_data} (loop_count={loop_count})")

    all_insights = state.get("insights", []) + result.get("insights", [])

    return {
        "insights": all_insights,
        "needs_more_data": needs_more_data,
        "follow_up_reason": result.get("follow_up_reason") if needs_more_data else None,
        "loop_count": loop_count + 1,
    }


# ===========================================================================
# Conditional routing function (used by graph.py, not itself a node)
# ===========================================================================
def route_after_analysis(state: AgentState) -> str:
    """
    Reads `needs_more_data` from state and returns the name of the next
    node to visit. This function is registered with LangGraph's
    `add_conditional_edges` and implements the "Need More Data? Yes/No"
    diamond from the workflow diagram.
    """
    if state.get("needs_more_data"):
        return "sql_generator"
    return "visualization"


# ===========================================================================
# AGENT 5: Visualization Agent
# ===========================================================================
def visualization_agent(state: AgentState) -> dict:
    """
    Builds chart(s) from the most recent retrieved data using pandas +
    matplotlib, and saves them as PNG files. This agent uses the LLM only
    to decide WHICH columns are most meaningful to plot and HOW (chart
    type, axis labels) -- the actual chart rendering is deterministic code.
    """
    data = state.get("raw_data", [])
    intent = state["intent"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not data:
        return {"chart_paths": [], "chart_descriptions": ["No data available to visualize."]}

    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    system_prompt = (
        "You are a data visualization expert. Given the available columns "
        "in a dataframe and the analytical goal, choose ONE good chart to "
        "visualize the data. "
        "Respond ONLY with a valid JSON object, no markdown, no extra text. "
        'Format: {"chart_type": "<bar|line>", "x_column": "<column name>", '
        '"y_column": "<column name>", "title": "<chart title>"}'
    )
    user_prompt = (
        f"Analytical goal: {intent}\n"
        f"Categorical/text columns available: {categorical_cols}\n"
        f"Numeric columns available: {numeric_cols}\n"
        f"Row count: {len(df)}\n\n"
        "Pick the most informative column pair and chart type for this goal."
    )

    try:
        choice = _call_llm_for_json(system_prompt, user_prompt)
        x_col = choice.get("x_column")
        y_col = choice.get("y_column")
        chart_type = choice.get("chart_type", "bar")
        title = choice.get("title", "Sales Analysis")

        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("LLM chose columns not present in the dataframe.")

        # Aggregate if x_column has repeated values (common after GROUP BY
        # queries that still have duplicate labels across loop iterations)
        plot_df = df.groupby(x_col, as_index=False)[y_col].sum()
        plot_df = plot_df.sort_values(by=y_col, ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(9, 5))
        if chart_type == "line":
            ax.plot(plot_df[x_col], plot_df[y_col], marker="o")
        else:
            ax.bar(plot_df[x_col], plot_df[y_col], color="#4C72B0")

        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        loop_count = state.get("loop_count", 0)
        # chart_path = os.path.join(OUTPUT_DIR, f"chart_{loop_count}.png")
        timestamp = datetime.now().strftime("%d-%b-%Y_%Hh%Mm")
        chart_path = os.path.join(OUTPUT_DIR, f"chart_{timestamp}_{loop_count}.png")

        plt.savefig(chart_path)
        plt.close(fig)

        description = f"{title} ({chart_type} chart of {y_col} by {x_col})"
        print(f"[Visualization] saved chart: {chart_path} | {description}")

        existing_paths = state.get("chart_paths", [])
        existing_descs = state.get("chart_descriptions", [])

        return {
            "chart_paths": existing_paths + [chart_path],
            "chart_descriptions": existing_descs + [description],
        }

    except Exception as e:
        print(f"[Visualization] skipped chart due to: {e}")
        return {
            "chart_paths": state.get("chart_paths", []),
            "chart_descriptions": state.get("chart_descriptions", []) + [f"Chart generation skipped: {e}"],
        }


# ===========================================================================
# AGENT 6: Report Writer
# ===========================================================================
def report_writer_agent(state: AgentState) -> dict:
    """
    Synthesizes everything gathered by the previous agents -- the original
    question, intent, all accumulated insights, SQL queries used, and chart
    descriptions -- into one polished, decision-ready Markdown report.

    The report's "Key Findings" section now explicitly includes a
    "Visualizations" subsection that describes what each generated chart
    shows and how it supports the findings, and the actual chart image(s)
    are embedded into the Markdown report (not just described in text),
    so the report is a self-contained artifact rather than something that
    needs the Streamlit UI open alongside it to make sense.

    Note: the embedded image path is relative (e.g. "outputs/chart_0.png").
    This renders correctly in the Streamlit app (st.markdown resolves it
    against the app's working directory) and in most editors/viewers that
    are opened from the project root. If you move final_report.md to a
    different folder without its outputs/ directory alongside it, the
    image references will break -- this is a known, accepted limitation
    for this POC rather than something handled with absolute paths or
    embedded base64 images.

    This is the final node in the graph; its output (`final_report`) is
    what gets shown to the business user.
    """
    question = state["question"]
    intent = state["intent"]
    insights = state.get("insights", [])
    sql_history = state.get("sql_history", [])
    chart_paths = state.get("chart_paths", [])
    chart_descriptions = state.get("chart_descriptions", [])

    has_charts = bool(chart_paths) and any(
        os.path.exists(p) for p in chart_paths
    )

    charts_block = (
        "\n".join(f"- {d}" for d in chart_descriptions)
        if chart_descriptions
        else "No charts were generated for this analysis."
    )

    system_prompt = (
        "You are a senior business analyst writing a final report for a "
        "business stakeholder. Write in clear, confident, plain language. "
        "Use the provided insights to construct a coherent narrative with "
        "a clear conclusion and 1-2 actionable recommendations. "
        "Output well-formatted Markdown with headings. Do not use emojis. "
        "Structure the report with these sections, in this order: "
        "Summary, Key Findings, Visualizations, Recommendation. "
        "Under Key Findings, list the concrete data-driven insights. "
        "Under Visualizations, write 1-2 sentences PER CHART explaining "
        "what it shows and what it confirms or reveals about the findings "
        "above -- do not just repeat the chart title, interpret it. "
        "Do not include the actual image markup yourself; just describe "
        "the chart(s) in that section, the image will be inserted "
        "automatically after your text."
    )
    user_prompt = (
        f"Original business question: \"{question}\"\n\n"
        f"Analytical focus: {intent}\n\n"
        f"Accumulated insights from data analysis:\n"
        + "\n".join(f"- {i}" for i in insights)
        + f"\n\nCharts generated (description of each):\n{charts_block}\n\n"
        "Write the final report now."
    )

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    report_body = response.content.strip()

    # ------------------------------------------------------------------
    # Embed the actual chart image(s) into the Markdown report so the
    # report is self-contained. We insert them right after the
    # "Visualizations" heading the LLM was instructed to write, rather
    # than appending them at the very end, so each image sits next to the
    # text that describes it. If for some reason the LLM did not include
    # that heading (model variance), we fall back to appending a
    # "Visualizations" section ourselves before the appendix.
    # ------------------------------------------------------------------
    image_markup = "\n\n".join(
        f"![{desc}]({path})\n\n*{desc}*"
        for path, desc in zip(chart_paths, chart_descriptions)
        if os.path.exists(path)
    )

    if has_charts and image_markup:
        heading_pattern = re.compile(r"^#{1,3}\s*Visualizations\s*$", re.MULTILINE)
        match = heading_pattern.search(report_body)
        if match:
            insert_at = match.end()
            report_body = (
                report_body[:insert_at]
                + "\n\n"
                + image_markup
                + report_body[insert_at:]
            )
        else:
            # Fallback: the LLM didn't produce the expected heading verbatim;
            # append a Visualizations section with the images ourselves so
            # the charts are never silently dropped from the report.
            report_body += "\n\n## Visualizations\n\n" + image_markup
    elif not has_charts:
        report_body += "\n\n## Visualizations\n\nNo charts were generated for this analysis."

    # Append a technical appendix so your manager can see the actual SQL
    # the system generated -- great for demonstrating transparency/auditability.
    appendix_lines = ["", "---", "", "## Appendix: Queries Executed"]
    for i, sql in enumerate(sql_history, start=1):
        appendix_lines.append(f"\n**Query {i}:**\n```sql\n{sql}\n```")

    final_report = report_body + "\n".join(appendix_lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # report_path = os.path.join(OUTPUT_DIR, "final_report.md")
    timestamp = datetime.now().strftime("%d-%b-%Y_%Hh%Mm")
    report_path = os.path.join(OUTPUT_DIR, f"final_report_{timestamp}.md")
    with open(report_path, "w") as f:
        f.write(final_report)

    print(f"[Report Writer] report saved to: {report_path}")

    return {"final_report": final_report}