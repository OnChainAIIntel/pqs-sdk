"""
PQS + AG2 (AutoGen) Example: Add prompt quality scoring to any AG2 agent.

PQS registers as a native AG2 tool — agents can call it before firing
any LLM prompt, catching weak inputs before they waste tokens.

Install:
    pip install prompt-quality-score autogen ag2

Get a free API key:
    https://pqs.onchainintel.net
"""

from autogen import AssistantAgent, UserProxyAgent, LLMConfig
from pqs_sdk import register_pqs_tools

# ── Step 1: Set up your AG2 agents as normal ─────────────────────────────────
llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")

assistant = AssistantAgent(
    name="assistant",
    system_message=(
        "You are a helpful AI assistant. Before responding to any task, "
        "use the score_prompt tool to check the quality of the incoming prompt. "
        "If it scores below 20/40, use optimize_prompt to improve it first."
    ),
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding", "use_docker": False},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=3,
)

# ── Step 2: Register PQS tools — one line ────────────────────────────────────
assistant, user_proxy = register_pqs_tools(
    assistant=assistant,
    user_proxy=user_proxy,
    api_key="your-pqs-api-key",  # free at pqs.onchainintel.net
    vertical="software"
)

# ── Step 3: Run as normal — agent now scores prompts before proceeding ────────
if __name__ == "__main__":
    user_proxy.initiate_chat(
        assistant,
        message="Write a Python function that sorts a list"
    )
    # Agent will:
    # 1. Score the prompt with PQS → likely D or F (too vague)
    # 2. Optimize it → rewrites to score 60+
    # 3. Use the optimized prompt for the actual task
