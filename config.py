"""
config.py
---------
Central place for configuration: API keys, LLM instantiation, and constants
shared across the project.

Why this file exists on its own:
    Every agent in agents.py needs an LLM instance. Rather than each agent
    creating its own ChatGroq client (and risking inconsistent settings),
    we create ONE shared instance here and import it everywhere. If you
    ever want to switch model, provider, or temperature, you change it
    in exactly one place.
"""

import os
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Groq API key
# ---------------------------------------------------------------------------
# Set this as an environment variable before running, e.g.:
#   export GROQ_API_KEY="your-key-here"        (Linux / Mac)
#   setx GROQ_API_KEY "your-key-here"           (Windows)
#
# We deliberately do NOT hardcode the key in source code -- that is a
# security best practice you should be able to speak to if asked.
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set. "
        "Run: export GROQ_API_KEY='your-key-here' before launching the POC."
    )

# ---------------------------------------------------------------------------
# Shared LLM instance used by every agent.
# temperature=0.1 keeps outputs deterministic and consistent, which matters
# a lot for an agent that must generate valid SQL or structured JSON.
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=GROQ_API_KEY,
)

# ---------------------------------------------------------------------------
# Other shared constants
# ---------------------------------------------------------------------------
DB_PATH = "sales_data.db"

# Safety cap: how many times the graph is allowed to loop back from the
# Analysis Agent to the SQL Generator before we force it to stop and report
# with whatever data it has. Prevents infinite loops if the LLM keeps
# asking for "more data" indefinitely.
MAX_REFINEMENT_LOOPS = 2

OUTPUT_DIR = "outputs"
