"""
app.py
------
Streamlit UI for the Autonomous Data Analytics Agent POC.

This file is purely a presentation layer. It contains NO agent logic of
its own -- it only:
  1. Calls build_database() / build_graph() from the existing modules.
  2. Streams the compiled graph's execution (via app.stream(), not
     app.invoke()) so each agent's progress can be displayed as it happens.
  3. Renders the final report, charts, and a separate chat interface.

This keeps the same separation-of-concerns principle as the rest of the
project: graph.py and agents.py still define WHAT happens; app.py only
decides HOW it is displayed.

Two independent sections, switchable via a horizontal selector:
  - "Analytics Report Workflow": the original six-agent pipeline from
    main.py, now with live per-agent progress instead of console prints.
  - "Chat With Data": the separate, lightweight agent from chat_agent.py
    for ad-hoc natural-language exploration of the same dataset. This
    does NOT go through the six-agent pipeline and does NOT produce a
    report -- it is a different, simpler agentic loop, kept deliberately
    separate (see chat_agent.py's docstring for why).

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

from database import build_database, DB_PATH
from state import AgentState

# ---------------------------------------------------------------------------
# Human-readable labels for each node, used when rendering live progress.
# Keeping this mapping here (not in graph.py) keeps graph.py free of any
# display-related concerns.
# ---------------------------------------------------------------------------
NODE_LABELS = {
    "intent_analyzer": "Intent Analyzer Agent",
    "sql_generator": "SQL Generator Agent",
    "data_retrieval": "Data Retrieval Agent",
    "analysis": "Analysis Agent",
    "visualization": "Visualization Agent",
    "report_writer": "Report Writer Agent",
}

st.set_page_config(
    page_title="Autonomous Data Analytics Agent",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar: environment status, shared across both tabs.
# ---------------------------------------------------------------------------
def render_sidebar():
    st.sidebar.title("Autonomous Data Analytics Agent")
    st.sidebar.caption("LangGraph multi-agent POC")

    st.sidebar.divider()
    st.sidebar.subheader("Environment")

    api_key_set = bool(os.environ.get("GROQ_API_KEY"))
    st.sidebar.write(
        "Groq API key: " + ("detected" if api_key_set else "NOT SET")
    )
    if not api_key_set:
        st.sidebar.error(
            "Set the GROQ_API_KEY environment variable before running "
            "queries, then restart the app."
        )

    db_exists = os.path.exists(DB_PATH)
    st.sidebar.write("Database: " + ("ready" if db_exists else "not yet created"))

    if st.sidebar.button("Rebuild synthetic database"):
        with st.spinner("Generating synthetic sales data (Jan-Dec 2025)..."):
            build_database(DB_PATH)
        st.sidebar.success("Database rebuilt.")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Model")
    st.sidebar.write("Provider: Groq")
    st.sidebar.write("Model: llama-3.3-70b-versatile")

    return api_key_set, db_exists


# ---------------------------------------------------------------------------
# Tab 1: the report-generation workflow, with live agent progress.
# ---------------------------------------------------------------------------
def render_workflow_tab(api_key_set: bool, db_exists: bool):
    st.header("Analytics Report Workflow")
    st.caption(
        "Ask a business question. The system interprets it, writes its own "
        "SQL, retrieves data, analyzes it, optionally loops back for a "
        "follow-up query, then produces a chart and a final report."
    )

    question = st.text_input(
        "Business question",
        value="Why did sales drop in the West region in March?",
        disabled=not (api_key_set and db_exists),
    )

    run_clicked = st.button(
        "Run analysis",
        type="primary",
        disabled=not (api_key_set and db_exists),
    )

    if not db_exists:
        st.info("Create the database from the sidebar before running.")
        return
    if not api_key_set:
        st.info("Set GROQ_API_KEY before running.")
        return

    if not run_clicked:
        return

    # Import here (not at module top) so app.py can still render the page
    # and show sidebar warnings even if GROQ_API_KEY is missing -- importing
    # config.py raises immediately if the key is absent, by design.
    from graph import build_graph

    app_graph = build_graph()
    initial_state: AgentState = {"question": question}

    st.divider()
    st.subheader("Execution Progress")

    status_blocks = {}
    progress_container = st.container()

    accumulated_state = dict(initial_state)
    loop_pass = 1

    with progress_container:
        for step in app_graph.stream(initial_state, stream_mode="updates"):
            # `step` is a dict like {"node_name": {state_updates}}
            for node_name, update in step.items():
                label = NODE_LABELS.get(node_name, node_name)

                # If this node has already run once before in this session
                # (a loop-back), label it clearly as a new pass.
                block_key = f"{node_name}_{loop_pass}"
                if node_name == "sql_generator" and block_key in status_blocks:
                    loop_pass += 1
                    block_key = f"{node_name}_{loop_pass}"

                if block_key not in status_blocks:
                    # Only the three nodes that actually repeat on a
                    # loop-back (sql_generator, data_retrieval, analysis)
                    # get the "(pass N)" suffix. visualization/report_writer
                    # always run exactly once, after looping is done, so
                    # labeling them with a pass number would be misleading.
                    loop_nodes = {"sql_generator", "data_retrieval", "analysis"}
                    suffix = (
                        f" (pass {loop_pass})"
                        if loop_pass > 1 and node_name in loop_nodes
                        else ""
                    )
                    status_blocks[block_key] = st.status(
                        f"{label}{suffix}", expanded=False
                    )

                status = status_blocks[block_key]
                _render_node_update(status, node_name, update)
                status.update(state="complete")

                # Merge this node's update into our running copy of state.
                # This mirrors what LangGraph does internally for simple
                # (non-reducer) fields -- the latest value for each key
                # overwrites the previous one, which matches how
                # AgentState's fields are used in agents.py (each agent
                # reads the full accumulated list/value and returns the
                # full new list/value, not a delta).
                accumulated_state.update(update)

    st.session_state["last_result"] = accumulated_state

    st.divider()
    st.subheader("Final Report")
    st.markdown(accumulated_state.get("final_report", "No report generated."))


def _render_node_update(status_block, node_name: str, update: dict):
    """
    Renders the relevant fields of a single node's state update inside its
    st.status() block. Each node returns different fields, so this just
    shows whatever is most relevant per node type.
    """
    if node_name == "intent_analyzer":
        status_block.write(f"Intent: {update.get('intent', '')}")
        status_block.write(f"Focus entities: {', '.join(update.get('focus_entities', []))}")

    elif node_name == "sql_generator":
        status_block.code(update.get("sql_query", ""), language="sql")

    elif node_name == "data_retrieval":
        status_block.write(f"Rows retrieved: {update.get('row_count', 0)}")
        data = update.get("raw_data", [])
        if data:
            status_block.dataframe(data[:10], use_container_width=True)

    elif node_name == "analysis":
        for insight in update.get("insights", []):
            status_block.write(f"- {insight}")
        if update.get("needs_more_data"):
            status_block.warning(
                f"Needs more data: {update.get('follow_up_reason', '')}"
            )
        else:
            status_block.write("Sufficient data gathered. Proceeding to visualization.")

    elif node_name == "visualization":
        for path, desc in zip(update.get("chart_paths", []), update.get("chart_descriptions", [])):
            status_block.write(desc)
            if os.path.exists(path):
                status_block.image(path)

    elif node_name == "report_writer":
        status_block.write("Report generated.")


# ---------------------------------------------------------------------------
# Tab 2: chat with the data directly (separate agentic loop, no report).
# ---------------------------------------------------------------------------
def render_chat_tab(api_key_set: bool, db_exists: bool, user_message: str | None):
    st.header("Chat With Data")
    st.caption(
        "Ask natural-language questions directly about the sales dataset. "
        "This is a separate, lightweight agent: it answers in a single "
        "conversational turn and does not produce a report. It can only "
        "answer questions about this dataset -- general questions are "
        "declined."
    )

    if not db_exists:
        st.info("Create the database from the sidebar before chatting.")
        return
    if not api_key_set:
        st.info("Set GROQ_API_KEY before chatting.")
        return

    from chat_agent import load_sales_dataframe, run_chat_turn

    if "chat_df" not in st.session_state:
        st.session_state["chat_df"] = load_sales_dataframe()
    if "chat_display_history" not in st.session_state:
        st.session_state["chat_display_history"] = []  # for rendering
    if "chat_lc_history" not in st.session_state:
        st.session_state["chat_lc_history"] = []  # LangChain message objects

    if st.button("Clear chat"):
        st.session_state["chat_display_history"] = []
        st.session_state["chat_lc_history"] = []
        st.rerun()

    # Render prior turns
    for turn in st.session_state["chat_display_history"]:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("tool_calls"):
                with st.expander("View data queries used"):
                    for call in turn["tool_calls"]:
                        st.code(call["expression"], language="python")
                        st.text(call["result"])

    if user_message:
        st.session_state["chat_display_history"].append(
            {"role": "user", "content": user_message}
        )
        with st.chat_message("user"):
            st.write(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Looking at the data..."):
                result = run_chat_turn(
                    user_message,
                    st.session_state["chat_lc_history"],
                    st.session_state["chat_df"],
                )
            st.write(result["answer"])
            if result["tool_calls"]:
                with st.expander("View data queries used"):
                    for call in result["tool_calls"]:
                        st.code(call["expression"], language="python")
                        st.text(call["result"])

        st.session_state["chat_lc_history"] = result["messages"]
        st.session_state["chat_display_history"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "tool_calls": result["tool_calls"],
            }
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    api_key_set, db_exists = render_sidebar()

    section = st.radio(
        "Section",
        # ["Analytics Report Workflow", "Chat With Data"],
        ["Chat With Data", "Analytics Report Workflow"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    user_message = None
    if section == "Chat With Data":
        user_message = st.chat_input("Ask a question about the sales data")

    if section == "Analytics Report Workflow":
        render_workflow_tab(api_key_set, db_exists)
    else:
        render_chat_tab(api_key_set, db_exists, user_message)


main()