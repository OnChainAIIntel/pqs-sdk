"""
PQS CrewAI Tool Integration
Native CrewAI tools for prompt quality scoring and optimization.

Add PQS as a quality gate to any CrewAI agent in 3 lines.
"""

from typing import Type, Optional
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Fallback base class when CrewAI is not installed
    class BaseTool:
        pass

from .client import PQSClient
from .models import ScoreResult, OptimizeResult


class PQSScoreInput(BaseModel):
    """Input schema for PQS Score Tool."""
    prompt: str = Field(
        ...,
        description="The prompt to score for quality before sending to an LLM"
    )
    vertical: str = Field(
        default="general",
        description=(
            "Domain vertical: software, content, business, "
            "education, science, crypto, or general"
        )
    )


class PQSOptimizeInput(BaseModel):
    """Input schema for PQS Optimize Tool."""
    prompt: str = Field(
        ...,
        description="The prompt to optimize for quality before sending to an LLM"
    )
    vertical: str = Field(
        default="general",
        description=(
            "Domain vertical: software, content, business, "
            "education, science, crypto, or general"
        )
    )


class PQSScoreTool(BaseTool):
    """
    CrewAI tool that scores a prompt before LLM inference.

    Add this to any agent as a pre-flight quality check.
    The agent will score the prompt and surface the result
    before deciding whether to proceed.

    Usage:
        from pqs_sdk import PQSScoreTool

        tool = PQSScoreTool(api_key="your-pqs-api-key")
        agent = Agent(
            role="Research Analyst",
            goal="Conduct quality research",
            tools=[tool]
        )
    """

    name: str = "PQS Prompt Quality Scorer"
    description: str = (
        "Scores a prompt for quality across 8 dimensions before sending it to an LLM. "
        "Returns a grade (A-F), score (0-40), and dimension breakdown. "
        "Use this BEFORE calling any LLM to ensure prompt quality. "
        "A score below 20/40 (grade D or F) indicates the prompt needs improvement."
    )
    args_schema: Type[BaseModel] = PQSScoreInput
    api_key: str = ""
    _client: Optional[PQSClient] = None

    def model_post_init(self, __context):
        self._client = PQSClient(api_key=self.api_key)

    def _run(self, prompt: str, vertical: str = "general") -> str:
        """Score the prompt and return a quality report."""
        try:
            result = self._client.score(prompt, vertical)
            output = (
                f"PQS Quality Score Report\n"
                f"========================\n"
                f"Score: {result.score}/40\n"
                f"Grade: {result.grade}\n"
                f"Verdict: {result.verdict}\n"
                f"Summary: {result.summary}\n"
            )
            if result.dimensions:
                output += "\nDimension Breakdown:\n"
                for dim, val in result.dimensions.items():
                    output += f"  {dim}: {val}\n"

            if not result.passed():
                output += (
                    f"\n⚠️  This prompt scored below passing threshold. "
                    f"Consider optimizing before proceeding."
                )
            return output
        except Exception as e:
            return f"PQS scoring error: {str(e)}"


class PQSOptimizeTool(BaseTool):
    """
    CrewAI tool that scores AND optimizes a prompt before LLM inference.

    When a prompt scores poorly, this tool rewrites it to score 60+/100.
    Costs $0.025 USDC per optimization via x402 on Base mainnet.

    Usage:
        from pqs_sdk import PQSOptimizeTool

        tool = PQSOptimizeTool(api_key="your-pqs-api-key")
        agent = Agent(
            role="Content Writer",
            goal="Write high quality content",
            tools=[tool]
        )
    """

    name: str = "PQS Prompt Quality Optimizer"
    description: str = (
        "Scores AND optimizes a prompt for quality before sending it to an LLM. "
        "Rewrites the prompt to score 60+ out of 100. "
        "Returns both the original and optimized prompt with scores. "
        "Use this when prompt quality needs improvement before LLM inference. "
        "Costs $0.025 USDC per optimization."
    )
    args_schema: Type[BaseModel] = PQSOptimizeInput
    api_key: str = ""
    _client: Optional[PQSClient] = None

    def model_post_init(self, __context):
        self._client = PQSClient(api_key=self.api_key)

    def _run(self, prompt: str, vertical: str = "general") -> str:
        """Optimize the prompt and return original + improved version."""
        try:
            result = self._client.optimize(prompt, vertical)
            output = (
                f"PQS Prompt Optimization Report\n"
                f"==============================\n"
                f"Original Score: {result.original_score}/40 ({result.original_grade})\n"
                f"Optimized Score: {result.optimized_score}/40 ({result.optimized_grade})\n"
                f"Improvement: +{result.improvement_delta()} points\n\n"
                f"Original Prompt:\n{result.original_prompt}\n\n"
                f"Optimized Prompt:\n{result.optimized_prompt}\n\n"
                f"Key Improvements:\n{result.improvements}"
            )
            return output
        except Exception as e:
            return f"PQS optimization error: {str(e)}"
