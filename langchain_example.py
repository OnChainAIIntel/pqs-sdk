"""
PQS + LangChain Example: Automatic prompt quality scoring with zero code changes.

PQSCallbackHandler hooks into LangChain's callback system and scores
every prompt automatically before it hits an LLM — no changes to your
existing chains or agents required.

Install:
    pip install prompt-quality-score langchain-openai

Get a free API key:
    https://pqs.onchainintel.net
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pqs_sdk import PQSCallbackHandler

# ── Step 1: Initialize the PQS callback handler ──────────────────────────────
handler = PQSCallbackHandler(
    api_key="your-pqs-api-key",
    vertical="software",   # software, content, business, education, science, crypto, general
    threshold=20,          # warn if score drops below this
    verbose=True           # print scores to console
)

# ── Step 2: Attach to any LangChain LLM — zero other changes needed ──────────
llm = ChatOpenAI(
    model="gpt-4o",
    callbacks=[handler]    # ← PQS added here
)

# ── Step 3: Use normally — PQS scores every prompt automatically ─────────────

# Weak prompt — will score low
print("--- Weak prompt ---")
response = llm.invoke([HumanMessage(content="help me write code")])
print(f"LLM response: {response.content[:100]}...")

# Strong prompt — will score higher
print("\n--- Strong prompt ---")
response = llm.invoke([HumanMessage(content=(
    "Write a Python function that takes a list of integers and returns "
    "the top 3 most frequent values. Include type hints, docstring, and "
    "handle edge cases like empty lists or ties. Return as a clean code block."
))])
print(f"LLM response: {response.content[:100]}...")

# ── Step 4: Review session summary ───────────────────────────────────────────
print(f"\n{handler.get_summary()}")
# → PQS Session Summary: 2 prompts scored | Avg score: 19.5/40 | Passed: 1/2
