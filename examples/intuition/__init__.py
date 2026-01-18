"""Intuition compression training utilities.

Provides problem dataset abstractions for distillation training.
"""
from examples.intuition.problems import (
    CodeProblemSet,
    JSONLProblemSet,
    MathProblemSet,
    Problem,
    ProblemSet,
)

__all__ = [
    "CodeProblemSet",
    "JSONLProblemSet",
    "MathProblemSet",
    "Problem",
    "ProblemSet",
]
