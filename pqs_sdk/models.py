"""PQS SDK data models."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ScoreResult:
    score: int
    grade: str
    verdict: str
    summary: str
    dimensions: Dict
    prompt: str
    vertical: str

    def passed(self) -> bool:
        return self.verdict.lower() == "pass"

    def __str__(self):
        return (
            f"PQS Score: {self.score}/40 | Grade: {self.grade} | "
            f"{self.verdict} | {self.summary}"
        )


@dataclass
class OptimizeResult:
    original_prompt: str
    optimized_prompt: str
    original_score: int
    optimized_score: int
    original_grade: str
    optimized_grade: str
    improvements: str
    dimensions: Dict

    def improvement_delta(self) -> int:
        return self.optimized_score - self.original_score

    def __str__(self):
        return (
            f"Optimized: {self.original_grade} ({self.original_score}/40) → "
            f"{self.optimized_grade} ({self.optimized_score}/40) "
            f"[+{self.improvement_delta()} pts]"
        )
