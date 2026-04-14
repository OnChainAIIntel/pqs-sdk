"""
PQS LangChain Callback Handler
Automatically scores every prompt before LLM inference.

Add PQS as a pre-flight quality gate to any LangChain chain or agent
with zero changes to your existing code.

Usage:
    from pqs_sdk import PQSCallbackHandler
    from langchain_openai import ChatOpenAI

    handler = PQSCallbackHandler(api_key="your-pqs-api-key", vertical="software")
    llm = ChatOpenAI(callbacks=[handler])

    # PQS now scores every prompt automatically before inference
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from .client import PQSClient

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        class BaseCallbackHandler:
            pass


class PQSCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler that scores prompts before LLM inference.

    Hooks into LangChain's callback system — fires automatically on every
    LLM call with zero changes to your existing chain or agent code.

    Usage:
        from pqs_sdk import PQSCallbackHandler
        from langchain_openai import ChatOpenAI

        handler = PQSCallbackHandler(
            api_key="your-pqs-api-key",
            vertical="software",   # optional, default: "general"
            threshold=20,          # optional, warn if score below this
            verbose=True           # optional, print scores to console
        )
        llm = ChatOpenAI(callbacks=[handler])

    Scores are available after each call via handler.last_score
    Full history available via handler.score_history
    """

    def __init__(
        self,
        api_key: str,
        vertical: str = "general",
        threshold: int = 20,
        verbose: bool = True,
        raise_on_fail: bool = False
    ):
        """
        Args:
            api_key: Your PQS API key (free at pqs.onchainintel.net)
            vertical: Domain vertical for scoring context
            threshold: Score below this triggers a warning (0-40)
            verbose: Print score results to console
            raise_on_fail: Raise exception if prompt scores below threshold
        """
        self.client = PQSClient(api_key=api_key)
        self.vertical = vertical
        self.threshold = threshold
        self.verbose = verbose
        self.raise_on_fail = raise_on_fail
        self.last_score = None
        self.score_history = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> None:
        """Fires before every LLM call — scores the prompt."""
        for prompt in prompts:
            self._score_prompt(prompt)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any
    ) -> None:
        """Fires before every chat model call — scores the messages."""
        for message_list in messages:
            combined = " ".join(
                m.content for m in message_list
                if hasattr(m, "content") and isinstance(m.content, str)
            )
            if combined.strip():
                self._score_prompt(combined)

    def _score_prompt(self, prompt: str) -> None:
        """Score a prompt and store the result."""
        try:
            result = self.client.score(prompt, self.vertical)
            self.last_score = result
            self.score_history.append(result)

            if self.verbose:
                status = "✅" if result.passed() else "⚠️"
                print(
                    f"{status} PQS: {result.score}/40 | "
                    f"Grade: {result.grade} | {result.summary}"
                )

            if self.raise_on_fail and not result.passed():
                raise ValueError(
                    f"PQS quality gate failed: score {result.score}/40 "
                    f"(grade {result.grade}) is below threshold {self.threshold}. "
                    f"Optimize your prompt before proceeding."
                )

        except ValueError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"PQS scoring error (non-blocking): {e}")

    def get_average_score(self) -> Optional[float]:
        """Returns average score across all scored prompts."""
        if not self.score_history:
            return None
        return sum(r.score for r in self.score_history) / len(self.score_history)

    def get_summary(self) -> str:
        """Returns a summary of all scored prompts in this session."""
        if not self.score_history:
            return "No prompts scored yet."
        avg = self.get_average_score()
        passed = sum(1 for r in self.score_history if r.passed())
        return (
            f"PQS Session Summary: {len(self.score_history)} prompts scored | "
            f"Avg score: {avg:.1f}/40 | "
            f"Passed: {passed}/{len(self.score_history)}"
        )
