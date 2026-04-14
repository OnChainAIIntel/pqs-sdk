"""
PQS SDK HTTP Client
Handles direct API communication with pqs.onchainintel.net
"""

import requests
from typing import Optional, Literal
from .models import ScoreResult, OptimizeResult

PQS_BASE_URL = "https://pqs.onchainintel.net"

VERTICALS = Literal[
    "software", "content", "business",
    "education", "science", "crypto", "general"
]


class PQSClient:
    """
    Direct HTTP client for the PQS API.

    Usage:
        from pqs_sdk import PQSClient

        client = PQSClient(api_key="your-key")
        result = client.score("Your prompt here", vertical="software")
        print(result)
    """

    def __init__(self, api_key: str, base_url: str = PQS_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": api_key
        })

    def score(
        self,
        prompt: str,
        vertical: str = "general"
    ) -> ScoreResult:
        """
        Score a prompt across 8 quality dimensions.

        Args:
            prompt: The prompt to score (max 10,000 chars)
            vertical: One of: software, content, business,
                      education, science, crypto, general

        Returns:
            ScoreResult with grade, score, and dimension breakdown

        Raises:
            ValueError: If prompt is empty or vertical is invalid
            requests.HTTPError: If API call fails
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.session.post(
            f"{self.base_url}/api/score",
            json={"prompt": prompt, "vertical": vertical}
        )
        response.raise_for_status()
        data = response.json()

        return ScoreResult(
            score=data.get("score", 0),
            grade=data.get("grade", "F"),
            verdict=data.get("verdict", "Fail"),
            summary=data.get("summary", ""),
            dimensions=data.get("dimensions", {}),
            prompt=prompt,
            vertical=vertical
        )

    def optimize(
        self,
        prompt: str,
        vertical: str = "general"
    ) -> OptimizeResult:
        """
        Score and optimize a prompt. Rewrites it to score 60+/100.
        Costs $0.025 USDC via x402 on Base mainnet.

        Args:
            prompt: The prompt to optimize
            vertical: One of: software, content, business,
                      education, science, crypto, general

        Returns:
            OptimizeResult with original + optimized prompt and scores
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = self.session.post(
            f"{self.base_url}/api/optimize",
            json={"prompt": prompt, "vertical": vertical}
        )
        response.raise_for_status()
        data = response.json()

        return OptimizeResult(
            original_prompt=prompt,
            optimized_prompt=data.get("optimizedPrompt", prompt),
            original_score=data.get("originalScore", 0),
            optimized_score=data.get("optimizedScore", 0),
            original_grade=data.get("originalGrade", "F"),
            optimized_grade=data.get("optimizedGrade", "F"),
            improvements=data.get("improvements", ""),
            dimensions=data.get("dimensions", {})
        )

    def check_health(self) -> bool:
        """Check if the PQS API is reachable."""
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            return response.status_code == 200
        except Exception:
            return False
