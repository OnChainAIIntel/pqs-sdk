"""
PQS (Prompt Quality Score) Python SDK
The world's first named AI prompt quality score.
Score, optimize, and compare prompts before LLM inference.

pip install pqs-sdk
"""

from .client import PQSClient
from .crewai_tool import PQSScoreTool, PQSOptimizeTool
from .langchain_callback import PQSCallbackHandler
from .models import ScoreResult, OptimizeResult

__version__ = "1.1.0"
__all__ = [
    "PQSClient",
    "PQSScoreTool",
    "PQSOptimizeTool",
    "PQSCallbackHandler",
    "ScoreResult",
    "OptimizeResult"
]
