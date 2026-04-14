"""
PQS + CrewAI Example: Add a prompt quality gate to your crew in 3 lines.

This example shows how to use PQS as a pre-flight quality gate
for any CrewAI agent. Before your agent sends a prompt to an LLM,
PQS scores it across 8 quality dimensions — catching weak inputs
before they waste tokens or produce bad outputs.

The AI input quality problem is real — and PQS named it.

Install:
    pip install pqs-sdk crewai

Get a free API key:
    https://pqs.onchainintel.net
"""

from crewai import Agent, Task, Crew
from pqs_sdk import PQSScoreTool, PQSOptimizeTool

# ── Step 1: Initialize PQS tools with your API key ──────────────────────────
pqs_score = PQSScoreTool(api_key="your-pqs-api-key")
pqs_optimize = PQSOptimizeTool(api_key="your-pqs-api-key")

# ── Step 2: Add PQS to any agent as a tool ──────────────────────────────────
researcher = Agent(
    role="Senior Research Analyst",
    goal="Conduct high-quality AI research with well-crafted prompts",
    backstory=(
        "You are a meticulous researcher who always ensures your prompts "
        "are high quality before sending them to any LLM. You use PQS to "
        "score and optimize every prompt before proceeding."
    ),
    tools=[pqs_score, pqs_optimize],  # ← PQS added here
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Produce clear, accurate technical content",
    backstory=(
        "You write technical content that is precise and well-structured. "
        "You always score your prompts with PQS before generating content "
        "to ensure maximum output quality."
    ),
    tools=[pqs_score],  # ← Score-only for writer
    verbose=True
)

# ── Step 3: Define tasks as normal ──────────────────────────────────────────
research_task = Task(
    description=(
        "Research the latest developments in AI agent frameworks in 2026. "
        "Before conducting research, use PQS to score and optimize your "
        "research prompt. Only proceed if the score is above 20/40."
    ),
    expected_output="A comprehensive summary of AI agent framework developments",
    agent=researcher
)

writing_task = Task(
    description=(
        "Write a technical blog post about AI agent frameworks based on the research. "
        "Use PQS to score your writing prompt before generating content."
    ),
    expected_output="A 500-word technical blog post ready for publication",
    agent=writer
)

# ── Step 4: Run the crew ─────────────────────────────────────────────────────
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print("\n=== CREW OUTPUT ===")
    print(result)
