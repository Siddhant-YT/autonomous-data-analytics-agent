"""
chat_agent.py
-------------
A separate, lightweight "chat with your data" agent. This is intentionally
DECOUPLED from the report-generation workflow in agents.py/graph.py -- it
does not produce a report, does not generate charts, and does not run
through the six-agent pipeline. It exists purely so a user can ask quick,
exploratory natural-language questions about the sales data and get a
direct, conversational answer back.

WHAT KIND OF "RAG" IS THIS, EXACTLY?
    Classic RAG retrieves chunks of unstructured TEXT from a vector store
    (embeddings + similarity search) and stuffs them into a prompt.
    There is no unstructured text here -- the source of truth is a
    structured SQL table. So this is NOT vector-based RAG.

    What this actually is: the LLM is given a TOOL that can query the
    `sales` table (via pandas), and it is allowed to call that tool,
    inspect the result, and decide whether it needs to call it again
    (e.g. to drill into a number it just saw) before answering. That
    decide -> act -> observe -> decide-again loop is what makes this
    "agentic" rather than a single fixed retrieve-then-answer step.
    The closest accepted term for this pattern is "agentic RAG over
    structured data" (sometimes called "Text-to-Pandas" or "Text-to-SQL
    agent" in industry literature) -- retrieval is still happening
    (we are retrieving relevant rows/aggregates rather than using the
    LLM's parametric memory), it is just retrieval FROM A DATAFRAME via
    a tool call, instead of retrieval FROM A VECTOR INDEX via embeddings.

WHY KEEP THIS SEPARATE FROM agents.py / graph.py:
    The report workflow is a fixed multi-stage pipeline with a loop and a
    final deliverable. This chat agent is a single conversational loop
    with no fixed number of stages and no report output. Mixing the two
    would make both harder to reason about. They share only the same
    underlying database and the same `llm` from config.py.

SCOPE GUARDRAIL:
    The system prompt explicitly restricts this agent to answering
    questions about the `sales` dataset only. General knowledge questions
    ("what is the capital of France?") are politely declined. This is
    enforced via prompting, plus a lightweight check described below.
"""

import json
import re

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config import llm, DB_PATH

import sqlite3


# ---------------------------------------------------------------------------
# Load the sales table into memory as a pandas DataFrame once, at import
# time. Re-loading on every question would be wasteful; the dataset is
# small enough (a few thousand rows) to comfortably hold in memory for the
# lifetime of the Streamlit session.
# ---------------------------------------------------------------------------
def load_sales_dataframe(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sales", conn)
    conn.close()
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


# ---------------------------------------------------------------------------
# The ONE tool this agent has access to: run a pandas query expressed as a
# Python expression string against the `sales` DataFrame, and return the
# result. Restricting the agent to a single, narrow, sandboxed tool (rather
# than e.g. arbitrary exec()) keeps this safe enough for a POC while still
# being genuinely agentic -- the LLM chooses WHEN to call it, WHAT
# expression to pass, and whether to call it again based on what comes back.
# ---------------------------------------------------------------------------
_SAFE_GLOBALS = {"pd": pd}


def _make_query_tool(df: pd.DataFrame):
    @tool
    def query_sales_data(pandas_expression: str) -> str:
        """
        Run a pandas expression against the sales DataFrame (named `df`)
        and return the result as text. The DataFrame has these columns:
        order_date (datetime64), region (str: North/South/East/West),
        product (str), category (str: Electronics/Apparel/Home/Beauty),
        units_sold (int), unit_price (float), revenue (float),
        discount_pct (float).

        Example expressions:
          "df.groupby('region')['revenue'].sum().sort_values(ascending=False)"
          "df[(df['category']=='Electronics') & (df['region']=='West')].groupby(df['order_date'].dt.month)['units_sold'].sum()"
          "df['revenue'].sum()"

        Always reference the dataframe as `df`. Return a pandas Series,
        DataFrame, or scalar -- it will be converted to text automatically.
        """
        try:
            # Restrict eval to the dataframe + pandas only -- no builtins,
            # no file/network access, no arbitrary code execution.
            local_scope = {"df": df}
            result = eval(pandas_expression, {"__builtins__": {}, **_SAFE_GLOBALS}, local_scope)

            if isinstance(result, (pd.DataFrame, pd.Series)):
                return result.to_string()
            return str(result)
        except Exception as e:
            return f"Error executing expression: {e}. Try a simpler or corrected pandas expression."

    return query_sales_data


# ---------------------------------------------------------------------------
# System prompt: this is what enforces the "data questions only" guardrail
# and tells the LLM how to use its one tool.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a data assistant that answers questions ONLY about \
the sales dataset described below. You are NOT a general-purpose assistant.

Dataset columns: order_date, region, product, category, units_sold, \
unit_price, revenue, discount_pct.
Data covers 2025-01-01 to 2025-12-31. Regions: North, South, East, West. \
Categories: Electronics, Apparel, Home, Beauty.

Rules:
1. If the question is about this sales dataset (trends, totals, comparisons, \
   breakdowns, specific records, etc.), use the query_sales_data tool to \
   compute the answer from real data. Never guess or fabricate numbers.
2. You may call the tool more than once if your first query was not specific \
   enough to answer confidently -- for example, if a total looks surprising, \
   break it down further before answering.
3. After getting a tool result, answer the user's question directly and \
   concisely in plain language, citing the actual numbers returned.
4. If the question is NOT about this sales dataset (general knowledge, \
   coding help, opinions, anything unrelated to sales/region/product/revenue), \
   politely decline and remind the user you can only answer questions about \
   the sales dataset.
5. Do not use emojis."""


MAX_TOOL_CALLS_PER_TURN = 4


def run_chat_turn(user_message: str, chat_history: list, df: pd.DataFrame) -> dict:
    """
    Runs one full agentic turn: sends the user's message (plus prior
    conversation history) to the LLM with the query_sales_data tool bound,
    lets the LLM call the tool zero or more times (up to a safety cap),
    and returns the final natural-language answer.

    Parameters
    ----------
    user_message : the new question from the user
    chat_history : list of prior LangChain message objects (Human/AI), so
                   the agent has conversational context across turns
    df            : the sales DataFrame loaded via load_sales_dataframe()

    Returns
    -------
    dict with keys:
        "answer"       : final text answer to show the user
        "tool_calls"   : list of {"expression": str, "result": str} actually
                          executed this turn (shown in the UI as a
                          transparency/debug trail)
        "messages"     : updated message list to store as the new chat_history
    """
    query_tool = _make_query_tool(df)
    llm_with_tools = llm.bind_tools([query_tool])

    messages = [SystemMessage(content=_SYSTEM_PROMPT)] + chat_history + [HumanMessage(content=user_message)]

    tool_call_log = []

    for _ in range(MAX_TOOL_CALLS_PER_TURN):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # No tool call requested -- this is the final answer.
            return {
                "answer": response.content,
                "tool_calls": tool_call_log,
                "messages": messages[1:],  # drop the system message before storing history
            }

        # The LLM requested one or more tool calls -- execute each and feed
        # the result back as a ToolMessage, then loop so the LLM can decide
        # whether it now has enough information to answer or needs another
        # query.
        for call in response.tool_calls:
            expression = call["args"].get("pandas_expression", "")
            result = query_tool.invoke({"pandas_expression": expression})
            tool_call_log.append({"expression": expression, "result": result})
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    # Safety cap reached without a final answer -- ask the LLM to summarize
    # with whatever it has gathered so far instead of looping forever.
    messages.append(
        HumanMessage(
            content="Please provide your best answer now based on the data gathered so far."
        )
    )
    final_response = llm.invoke(messages)
    return {
        "answer": final_response.content,
        "tool_calls": tool_call_log,
        "messages": messages[1:] + [final_response],
    }
