# Autonomous Data Analytics Agent (LangGraph POC)

## 1. Overview

This POC implements an autonomous, multi-agent data analytics system using
LangGraph. It has two independent capabilities, both built on top of the
same synthetic sales dataset:

1. **Analytics Report Workflow** -- a six-agent LangGraph pipeline that
   takes a natural-language business question (e.g. "Why did sales drop
   in the West region in March?"), autonomously writes and executes SQL,
   analyzes the results, optionally loops back for follow-up queries, and
   produces a chart plus a final written report.

2. **Chat With Data** -- a separate, lightweight conversational agent for
   ad-hoc exploration of the same dataset. It does not run through the
   six-agent pipeline and does not produce a report; it answers direct
   questions about the data in natural language, backed by live pandas
   computation rather than memorized or fabricated numbers.

Both capabilities are exposed through a single Streamlit application
(`app.py`), as two tabs. 

---

## 2. Architecture and Workflow

### 2.1 Analytics Report Workflow

```
Business Question
       |
Intent Analyzer Agent
       |
SQL Generator Agent   <------------------------+
       |                                       |
Data Retrieval Agent                           |
       |                                       |
Analysis Agent                                 |
       |                                       |
  Need More Data? --- Yes ----------------------+
       |
       No
       |
Visualization Agent
       |
Report Writer Agent
       |
     Output
```

This is implemented as a LangGraph `StateGraph` (`graph.py`). All
agents read from and write to one shared `AgentState` (a `TypedDict`
defined in `state.py`) that is passed through the graph. The only
conditional edge in the graph is `route_after_analysis()`, which inspects
`needs_more_data` in the state and returns either `"sql_generator"`
(loop back for a follow-up query) or `"visualization"` (proceed). This is
what creates the cycle in the graph: `analysis -> sql_generator` is a real
loop-back edge, not just a linear pipeline.

A loop counter (`loop_count` in state) is capped by `MAX_REFINEMENT_LOOPS`
(set in `config.py`) so the graph cannot loop indefinitely even if the LLM
keeps requesting more data.

The compiled graph supports execution mode, defined once in
`graph.py` and used by different entry points:
- `app.stream(state, stream_mode="updates")` -- yields the state update
  after every node finishes, one node at a time. Used by `app.py`
  (Streamlit UI) to render live per-agent progress.

### 2.2 Chat With Data

```
User question
       |
LLM (with one bound tool: query_sales_data)
       |
  Tool call requested? --- Yes --> Execute pandas expression against
       |    ^                      the in-memory sales DataFrame
       |    |                              |
       |    +------------------------------+
       No   (LLM may call the tool again if the
       |     first result wasn't sufficient)
       |
Final natural-language answer
```

This is a single conversational loop, implemented in `chat_agent.py`,
entirely separate from the `StateGraph` used by the report workflow. The
LLM is bound to one tool (`query_sales_data`) via `bind_tools()`. On each
turn, the LLM decides whether it needs to call the tool, executes a
pandas expression against the `sales` DataFrame if so, observes the
result, and may call the tool again (up to `MAX_TOOL_CALLS_PER_TURN`)
before producing a final answer. Conversation history is preserved across
turns within a Streamlit session.

**Why this is "agentic RAG over structured data" and not classic RAG:**
Classic RAG retrieves chunks of unstructured text from a vector store via
embeddings and similarity search. There is no unstructured text and no
vector store here -- the source of truth is a structured table. Retrieval
still happens (the LLM is retrieving relevant rows/aggregates rather than
relying on parametric memory), but it happens through a tool call against
a pandas DataFrame rather than a similarity search against an embedding
index. It is "agentic" specifically because the LLM controls the
retrieve -> observe -> decide-to-retrieve-again loop itself, rather than
following a fixed single retrieve-then-answer step.

This module is kept separate from `agents.py` / `graph.py` deliberately:
the report workflow is a fixed multi-stage pipeline with a defined
deliverable (a report); the chat agent is an open-ended conversational
loop with no fixed number of stages and no report output. They share only
the same `llm` instance (`config.py`) and underlying database.

---

## 3. Agents and Responsibilities

### Report workflow agents (`agents.py`)

| Agent | Function | LLM call? | Responsibility |
|---|---|---|---|
| Intent Analyzer | `intent_analyzer_agent` | Yes | Interprets the raw business question into a structured analytical goal and a list of focus entities (region, time period, category, metric). |
| SQL Generator | `sql_generator_agent` | Yes | Writes a SQLite query against the `sales` table. On loop-back passes, writes a more targeted follow-up query based on the Analysis Agent's stated reason. |
| Data Retrieval | `data_retrieval_agent` | No | Executes the SQL query against `sales_data.db` and returns rows as a list of dicts. Deliberately non-LLM: a pure, deterministic execution step. Catches SQL errors and returns an empty result instead of crashing. |
| Analysis | `analysis_agent` | Yes | Extracts concrete insights from the retrieved data sample and decides `needs_more_data: true/false`. This decision drives the conditional loop-back edge. Enforces the `MAX_REFINEMENT_LOOPS` safety cap. |
| Visualization | `visualization_agent` | Yes (column/chart choice only) | Chooses which columns and chart type best represent the findings via the LLM, then renders the chart deterministically with matplotlib/pandas. Saves PNG to `outputs/`. |
| Report Writer | `report_writer_agent` | Yes | Synthesizes all accumulated insights into a final Markdown report (Summary, Key Findings, Recommendation) plus a technical appendix listing every SQL query executed. Saves to `outputs/final_report.md`. |

Routing function (not an LLM-calling agent, used by `graph.py`):
- `route_after_analysis(state)` -- reads `needs_more_data` from state and
  returns the name of the next node (`"sql_generator"` or
  `"visualization"`).

### Chat agent (`chat_agent.py`)

| Component | Responsibility |
|---|---|
| `load_sales_dataframe()` | Loads the full `sales` table from SQLite into a pandas DataFrame once per session. |
| `query_sales_data` (tool) | The agent's only tool. Executes a pandas expression string against the DataFrame inside a restricted `eval()` (no builtins, no file/network access) and returns the result as text. |
| `run_chat_turn()` | Runs one full agentic turn: binds the tool to the LLM, lets the LLM call it zero or more times, and returns the final answer plus a log of every tool call made (for transparency in the UI). |

---

## 4. Data Source

`database.py` generates a synthetic SQLite database (`sales_data.db`) on
first run. There is no external data dependency.

**Schema (table: `sales`):**

| Column | Type | Notes |
|---|---|---|
| id | INTEGER | Primary key |
| order_date | TEXT (`YYYY-MM-DD`) | Spans 2025-01-01 to 2025-12-31 |
| region | TEXT | One of: North, South, East, West |
| product | TEXT | e.g. Laptop, T-Shirt, Blender, Shampoo |
| category | TEXT | One of: Electronics, Apparel, Home, Beauty |
| units_sold | INTEGER | |
| unit_price | REAL | |
| revenue | REAL | `units_sold * unit_price * (1 - discount_pct)` |
| discount_pct | REAL | 0.0 - 0.30 |

**Two intentional anomalies are baked into the data generation logic**, so
both the report workflow and the chat agent have real, discoverable
patterns to work with rather than pure noise:

1. **West region, Electronics category, March 2025** -- units sold drop
   to roughly 25% of the normal baseline, with elevated discounting
   (15-30%), simulating a supply disruption or competitor issue.
2. **All regions, all categories, November-December 2025** -- units sold
   increase to roughly 1.5x-1.9x the normal baseline, simulating a
   holiday/year-end seasonal demand surge.

`get_schema_description()` in `database.py` returns a plain-text
description of the schema (columns, types, value ranges) that is injected
into the SQL Generator Agent's prompt, so the schema is defined in exactly
one place and never hardcoded into agent prompts directly.

To regenerate the database (e.g. after changing the generation logic),
delete `sales_data.db` and rerun `app.py` (which
auto-create it if missing) or run `python database.py` directly, or use
the "Rebuild synthetic database" button in the Streamlit sidebar.

---

## 5. Streamlit Application Usage

Run with:
```bash
streamlit run app.py
```

The app has a sidebar (shared across both tabs) and two tabs.

### Sidebar
- Shows whether `GROQ_API_KEY` is set and whether `sales_data.db` exists.
- "Rebuild synthetic database" button regenerates `sales_data.db` from
  scratch using the current generation logic in `database.py`.
- Displays the active model provider/name.

### Tab: Analytics Report Workflow
- Text input for the business question (pre-filled with a default
  example).
- "Run analysis" button triggers `graph.build_graph().stream(...)`.
- As the graph executes, one `st.status` block appears per agent, in
  real time, showing that agent's relevant output:
  - Intent Analyzer: extracted intent and focus entities.
  - SQL Generator: the generated SQL query.
  - Data Retrieval: row count and a preview table of retrieved rows.
  - Analysis: extracted insights, and whether a follow-up query is
    needed (with the reason, if so).
  - Visualization: the chart image and its description.
  - Report Writer: confirmation that the report was generated.
- If the Analysis Agent triggers a loop-back, the SQL Generator, Data
  Retrieval, and Analysis status blocks reappear labeled "(pass 2)" so
  the refinement loop is visually distinguishable from the first pass.
- The final Markdown report is rendered below the progress section once
  the graph reaches `END`.

### Tab: Chat With Data
- Standard chat interface (`st.chat_input` / `st.chat_message`).
- Each assistant response includes an expandable "View data queries
  used" section showing the exact pandas expression(s) executed and
  their raw results, for transparency into what the agent actually
  computed.
- "Clear chat" button resets the conversation history.
- Out-of-scope questions (anything not about the `sales` dataset) are
  declined by the agent directly, per its system prompt -- this is not
  filtered by the UI layer.

---

## 6. Setup and Execution

### Requirements
```bash
pip install -r requirements.txt
```

### Environment variable
```bash
export GROQ_API_KEY="your-key-here"        # Linux / Mac
setx GROQ_API_KEY "your-key-here"          # Windows
```

`config.py` raises an error immediately on import if `GROQ_API_KEY` is
not set.

### Run the Streamlit app (recommended)
```bash
streamlit run app.py
```


### Project files

```
.
├── config.py        Shared ChatGroq LLM instance, API key loading, constants
├── state.py          AgentState definition (shared state for the report workflow)
├── database.py       Synthetic SQLite sales database generator + schema description
├── agents.py          Six report-workflow agents + the conditional routing function
├── graph.py           Builds and compiles the LangGraph StateGraph
├── chat_agent.py      Separate agentic chat-with-data loop (pandas tool + LLM)
├── app.py             Streamlit UI: report workflow tab + chat tab
├── requirements.txt
└── outputs/           Generated charts (.png) and final_report.md (created at runtime)
```
