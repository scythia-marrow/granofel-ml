"""Problem dataset abstractions for distillation training.

Provides interfaces for problem sources and solution validation.
Used to generate training examples for the compression model.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
import json
import re
import subprocess


class Problem:
    """A single problem with its metadata."""

    def __init__(
        self,
        prompt: str,
        answer: Any = None,
        metadata: Optional[dict] = None
    ):
        self.prompt = prompt
        self.answer = answer
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "answer": self.answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Problem":
        return cls(
            prompt=data["prompt"],
            answer=data.get("answer"),
            metadata=data.get("metadata", {}),
        )


class ProblemSet(ABC):
    """Abstract base class for problem datasets.

    Implementations must provide:
    - __iter__: Yields Problem instances
    - validate: Checks if a solution is correct for a given problem
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Problem]:
        """Yield problems from the dataset."""
        raise NotImplementedError("Subclasses must implement __iter__")

    @abstractmethod
    def validate(self, problem: Problem, solution: str) -> bool:
        """Check if the solution is correct for the given problem.

        Args:
            problem: The problem being solved
            solution: The proposed solution string

        Returns:
            True if the solution is correct, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate")

    def __len__(self) -> int:
        """Return the number of problems. Override for efficiency."""
        return sum(1 for _ in self)


class MathProblemSet(ProblemSet):
    """Problem set for math problems with numeric answers.

    Validates by extracting numeric values from solutions and comparing
    to the expected answer within a tolerance.
    """

    def __init__(
        self,
        problems: list[Problem],
        tolerance: float = 1e-6
    ):
        self._problems = problems
        self._tolerance = tolerance

    def __iter__(self) -> Iterator[Problem]:
        yield from self._problems

    def __len__(self) -> int:
        return len(self._problems)

    def validate(self, problem: Problem, solution: str) -> bool:
        """Validate by comparing extracted numeric answer."""
        if problem.answer is None:
            return False

        extracted = self._extract_number(solution)
        if extracted is None:
            return False

        expected = float(problem.answer)
        return abs(extracted - expected) <= self._tolerance

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract the last number from text (common answer format)."""
        # Match integers, decimals, and scientific notation
        pattern = r"-?\d+\.?\d*(?:[eE][+-]?\d+)?"
        matches = re.findall(pattern, text)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    @classmethod
    def from_list(cls, items: list[tuple[str, float]], **kwargs) -> "MathProblemSet":
        """Create from list of (prompt, answer) tuples."""
        problems = [
            Problem(prompt=prompt, answer=answer)
            for prompt, answer in items
        ]
        return cls(problems, **kwargs)


class CodeProblemSet(ProblemSet):
    """Problem set for code problems with test case validation.

    Each problem contains test cases that are run against the solution.
    Supports Python code execution with configurable timeout.
    """

    def __init__(
        self,
        problems: list[Problem],
        timeout: float = 5.0
    ):
        self._problems = problems
        self._timeout = timeout

    def __iter__(self) -> Iterator[Problem]:
        yield from self._problems

    def __len__(self) -> int:
        return len(self._problems)

    def validate(self, problem: Problem, solution: str) -> bool:
        """Validate by running test cases against the solution code."""
        test_cases = problem.metadata.get("test_cases", [])
        if not test_cases:
            return False

        for test_case in test_cases:
            test_input = test_case.get("input", "")
            expected_output = test_case.get("output", "")

            # Construct test program
            test_code = f"{solution}\n\n{test_input}"

            try:
                result = subprocess.run(
                    ["python", "-c", test_code],
                    capture_output=True,
                    text=True,
                    timeout=self._timeout,
                )
                actual_output = result.stdout.strip()
                if actual_output != expected_output.strip():
                    return False
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                return False

        return True

    @classmethod
    def from_list(
        cls,
        items: list[tuple[str, list[dict]]],
        **kwargs
    ) -> "CodeProblemSet":
        """Create from list of (prompt, test_cases) tuples.

        Each test_case dict should have 'input' and 'output' keys.
        """
        problems = [
            Problem(prompt=prompt, metadata={"test_cases": test_cases})
            for prompt, test_cases in items
        ]
        return cls(problems, **kwargs)


class JSONLProblemSet(ProblemSet):
    """Problem set loaded from JSONL file with custom validator.

    Each line in the file should be a JSON object with at least a 'prompt' field.
    The validator function receives the problem and solution for custom validation.
    """

    def __init__(
        self,
        path: Path | str,
        validator: Callable[[Problem, str], bool],
    ):
        self._path = Path(path)
        self._validator = validator
        self._problems: Optional[list[Problem]] = None

    def _load(self) -> list[Problem]:
        """Lazy load problems from file."""
        if self._problems is not None:
            return self._problems

        self._problems = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._problems.append(Problem.from_dict(data))

        return self._problems

    def __iter__(self) -> Iterator[Problem]:
        yield from self._load()

    def __len__(self) -> int:
        return len(self._load())

    def validate(self, problem: Problem, solution: str) -> bool:
        """Validate using the custom validator function."""
        return self._validator(problem, solution)

    @classmethod
    def with_exact_match(cls, path: Path | str) -> "JSONLProblemSet":
        """Create with exact string match validation against problem.answer."""
        def exact_match(problem: Problem, solution: str) -> bool:
            if problem.answer is None:
                return False
            return solution.strip() == str(problem.answer).strip()

        return cls(path, exact_match)

    @classmethod
    def with_contains_match(cls, path: Path | str) -> "JSONLProblemSet":
        """Create with substring match validation (answer in solution)."""
        def contains_match(problem: Problem, solution: str) -> bool:
            if problem.answer is None:
                return False
            return str(problem.answer).strip() in solution

        return cls(path, contains_match)
