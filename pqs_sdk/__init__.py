"""
PQS (Prompt Quality Score) Python SDK
The world's first named AI prompt quality score.
Score, optimize, and compare prompts before LLM inference.

pip install prompt-quality-score
"""

from .client import PQSClient
from .crewai_tool import PQSScoreTool, PQSOptimizeTool
from .langchain_callback import PQSCallbackHandler
from .ag2_tool import register_pqs_tools, create_pqs_tool
from .models import ScoreResult, OptimizeResult

__version__ = "1.2.1"
__all__ = [
    "PQSClient",
    "PQSScoreTool",
    "PQSOptimizeTool",
    "PQSCallbackHandler",
    "register_pqs_tools",
    "create_pqs_tool",
    "ScoreResult",
    "OptimizeResult"
]
