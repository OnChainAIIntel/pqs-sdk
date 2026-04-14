"""
PQS AG2 (AutoGen) Tool Integration
Native AG2 tool registration for prompt quality scoring.

Registers PQS as a callable tool for any AG2/AutoGen agent,
allowing agents to score and optimize prompts before LLM inference.

Usage:
    from pqs_sdk import register_pqs_tools
    from autogen import AssistantAgent, UserProxyAgent, LLMConfig

    assistant, user_proxy = register_pqs_tools(
        assistant=AssistantAgent("assistant", llm_config=llm_config),
        user_proxy=UserProxyAgent("user_proxy", ...),
        api_key="your-pqs-api-key"
    )
"""

from typing import Optional, Tuple
from .client import PQSClient

try:
    from autogen import register_function, ConversableAgent
    AUTOGEN_AVAILABLE = True
except ImportError:
    try:
        from ag2 import register_function, ConversableAgent
        AUTOGEN_AVAILABLE = True
    except ImportError:
        AUTOGEN_AVAILABLE = False


def register_pqs_tools(
    assistant,
    user_proxy,
    api_key: str,
    vertical: str = "general",
    include_optimize: bool = True
):
    """
    Register PQS score and optimize tools with any AG2/AutoGen agent pair.

    Args:
        assistant: AG2 AssistantAgent instance
        user_proxy: AG2 UserProxyAgent instance
        api_key: Your PQS API key (free at pqs.onchainintel.net)
        vertical: Domain vertical for scoring context
        include_optimize: Also register the optimize tool ($0.025 USDC)

    Returns:
        Tuple of (assistant, user_proxy) with PQS tools registered

    Usage:
        assistant, user_proxy = register_pqs_tools(
            assistant=assistant,
            user_proxy=user_proxy,
            api_key="your-key",
            vertical="software"
        )
    """
    client = PQSClient(api_key=api_key)

    def score_prompt(prompt: str, vertical: str = vertical) -> str:
        """
        Score a prompt for quality across 8 dimensions before sending to an LLM.
        Returns grade (A-F), score (0-40), and improvement suggestions.
        Use this BEFORE generating any LLM response to ensure prompt quality.
        """
        try:
            result = client.score(prompt, vertical)
            output = (
                f"PQS Quality Score\n"
                f"Score: {result.score}/40 | Grade: {result.grade} | {result.verdict}\n"
                f"Summary: {result.summary}"
            )
            if not result.passed():
                output += "\n⚠️ Prompt quality is below passing threshold. Consider optimizing."
            return output
        except Exception as e:
            return f"PQS scoring error: {str(e)}"

    register_function(
        score_prompt,
        caller=assistant,
        executor=user_proxy,
        name="score_prompt",
        description=(
            "Score a prompt for quality before sending it to an LLM. "
            "Returns a grade (A-F) and 8-dimension quality breakdown. "
            "Use this pre-inference to catch weak prompts before they waste tokens."
        )
    )

    if include_optimize:
        def optimize_prompt(prompt: str, vertical: str = vertical) -> str:
            """
            Score AND optimize a prompt before sending to an LLM.
            Rewrites the prompt to score 60+ out of 100.
            Costs $0.025 USDC per optimization via x402 on Base mainnet.
            Returns both original and optimized prompt with scores.
            """
            try:
                result = client.optimize(prompt, vertical)
                return (
                    f"PQS Optimization Result\n"
                    f"Original: {result.original_grade} ({result.original_score}/40)\n"
                    f"Optimized: {result.optimized_grade} ({result.optimized_score}/40) "
                    f"[+{result.improvement_delta()} pts]\n\n"
                    f"Use this optimized prompt:\n{result.optimized_prompt}"
                )
            except Exception as e:
                return f"PQS optimization error: {str(e)}"

        register_function(
            optimize_prompt,
            caller=assistant,
            executor=user_proxy,
            name="optimize_prompt",
            description=(
                "Score AND optimize a prompt before sending it to an LLM. "
                "Rewrites the prompt to score 60+/100. "
                "Use this when prompt quality needs improvement before inference. "
                "Costs $0.025 USDC per optimization."
            )
        )

    return assistant, user_proxy


def create_pqs_tool(api_key: str, vertical: str = "general"):
    """
    Create a standalone PQS scoring function for use with AG2's FunctionTool.
    Compatible with autogen_core.tools.FunctionTool (Microsoft AutoGen v0.4+).

    Usage:
        from autogen_core.tools import FunctionTool
        from pqs_sdk import create_pqs_tool

        pqs_fn = create_pqs_tool(api_key="your-key", vertical="software")
        pqs_tool = FunctionTool(pqs_fn, description="Score prompt quality before LLM inference")
    """
    client = PQSClient(api_key=api_key)

    def score_prompt_tool(prompt: str) -> str:
        """Score a prompt for quality before LLM inference. Returns grade A-F and score 0-40."""
        try:
            result = client.score(prompt, vertical)
            return (
                f"Grade: {result.grade} | Score: {result.score}/40 | "
                f"{result.verdict} | {result.summary}"
            )
        except Exception as e:
            return f"PQS error: {str(e)}"

    return score_prompt_tool
