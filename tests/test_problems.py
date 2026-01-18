import pytest
import tempfile
from pathlib import Path

from granofel.problems import (
    Problem,
    ProblemSet,
    MathProblemSet,
    CodeProblemSet,
    JSONLProblemSet,
)


class TestProblem:
    def test_problem_creation(self):
        p = Problem(prompt="What is 2+2?", answer=4)
        assert p.prompt == "What is 2+2?"
        assert p.answer == 4
        assert p.metadata == {}

    def test_problem_with_metadata(self):
        p = Problem(prompt="test", answer="x", metadata={"difficulty": "easy"})
        assert p.metadata["difficulty"] == "easy"

    def test_problem_to_dict(self):
        p = Problem(prompt="test", answer=42, metadata={"key": "value"})
        d = p.to_dict()
        assert d["prompt"] == "test"
        assert d["answer"] == 42
        assert d["metadata"]["key"] == "value"

    def test_problem_from_dict(self):
        d = {"prompt": "test", "answer": 42, "metadata": {"key": "value"}}
        p = Problem.from_dict(d)
        assert p.prompt == "test"
        assert p.answer == 42
        assert p.metadata["key"] == "value"


class TestMathProblemSet:
    def test_iteration(self):
        ps = MathProblemSet.from_list([
            ("What is 2+2?", 4),
            ("What is 3*3?", 9),
        ])
        problems = list(ps)
        assert len(problems) == 2
        assert problems[0].prompt == "What is 2+2?"
        assert problems[1].answer == 9

    def test_len(self):
        ps = MathProblemSet.from_list([
            ("Q1", 1),
            ("Q2", 2),
            ("Q3", 3),
        ])
        assert len(ps) == 3

    def test_validate_correct_integer(self):
        ps = MathProblemSet.from_list([("What is 2+2?", 4)])
        problem = list(ps)[0]
        assert ps.validate(problem, "The answer is 4")
        assert ps.validate(problem, "4")
        assert ps.validate(problem, "I think it's 4.")

    def test_validate_correct_decimal(self):
        ps = MathProblemSet.from_list([("What is 1/3?", 0.333333)])
        problem = list(ps)[0]
        assert ps.validate(problem, "Approximately 0.333333")

    def test_validate_incorrect(self):
        ps = MathProblemSet.from_list([("What is 2+2?", 4)])
        problem = list(ps)[0]
        assert not ps.validate(problem, "The answer is 5")
        assert not ps.validate(problem, "I don't know")

    def test_validate_extracts_last_number(self):
        ps = MathProblemSet.from_list([("Calculate", 42)])
        problem = list(ps)[0]
        # Should extract the last number (42) not intermediate ones
        assert ps.validate(problem, "First I got 10, then 20, finally 42")

    def test_validate_with_tolerance(self):
        ps = MathProblemSet.from_list([("pi?", 3.14159)], tolerance=0.01)
        problem = list(ps)[0]
        assert ps.validate(problem, "About 3.14")
        assert not ps.validate(problem, "About 3.0")

    def test_validate_no_answer(self):
        p = Problem(prompt="test", answer=None)
        ps = MathProblemSet([p])
        assert not ps.validate(p, "42")

    def test_validate_scientific_notation(self):
        ps = MathProblemSet.from_list([("Large number", 1e6)])
        problem = list(ps)[0]
        assert ps.validate(problem, "The answer is 1e6")
        assert ps.validate(problem, "1000000")


class TestCodeProblemSet:
    def test_iteration(self):
        ps = CodeProblemSet.from_list([
            ("Write hello", [{"input": "print('hello')", "output": "hello"}]),
        ])
        problems = list(ps)
        assert len(problems) == 1
        assert problems[0].prompt == "Write hello"

    def test_validate_correct_code(self):
        ps = CodeProblemSet.from_list([
            ("Add function", [
                {"input": "print(add(2, 3))", "output": "5"},
                {"input": "print(add(0, 0))", "output": "0"},
            ]),
        ])
        problem = list(ps)[0]
        solution = "def add(a, b): return a + b"
        assert ps.validate(problem, solution)

    def test_validate_incorrect_code(self):
        ps = CodeProblemSet.from_list([
            ("Add function", [
                {"input": "print(add(2, 3))", "output": "5"},
            ]),
        ])
        problem = list(ps)[0]
        solution = "def add(a, b): return a - b"  # Wrong operation
        assert not ps.validate(problem, solution)

    def test_validate_syntax_error(self):
        ps = CodeProblemSet.from_list([
            ("Test", [{"input": "print(1)", "output": "1"}]),
        ])
        problem = list(ps)[0]
        solution = "def broken(:"  # Syntax error
        assert not ps.validate(problem, solution)

    def test_validate_timeout(self):
        ps = CodeProblemSet.from_list([
            ("Test", [{"input": "print(1)", "output": "1"}]),
        ], timeout=0.1)
        problem = list(ps)[0]
        solution = "import time; time.sleep(10)"  # Will timeout
        assert not ps.validate(problem, solution)

    def test_validate_no_test_cases(self):
        p = Problem(prompt="test", metadata={})
        ps = CodeProblemSet([p])
        assert not ps.validate(p, "print('hello')")


class TestJSONLProblemSet:
    def test_load_and_iterate(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Q1", "answer": "A1"}\n')
            f.write('{"prompt": "Q2", "answer": "A2"}\n')
            f.flush()
            path = Path(f.name)

        try:
            ps = JSONLProblemSet.with_exact_match(path)
            problems = list(ps)
            assert len(problems) == 2
            assert problems[0].prompt == "Q1"
            assert problems[1].answer == "A2"
        finally:
            path.unlink()

    def test_len(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Q1", "answer": "A1"}\n')
            f.write('{"prompt": "Q2", "answer": "A2"}\n')
            f.flush()
            path = Path(f.name)

        try:
            ps = JSONLProblemSet.with_exact_match(path)
            assert len(ps) == 2
        finally:
            path.unlink()

    def test_exact_match_validation(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Capital of France?", "answer": "Paris"}\n')
            f.flush()
            path = Path(f.name)

        try:
            ps = JSONLProblemSet.with_exact_match(path)
            problem = list(ps)[0]
            assert ps.validate(problem, "Paris")
            assert ps.validate(problem, "  Paris  ")  # Strips whitespace
            assert not ps.validate(problem, "The answer is Paris")
        finally:
            path.unlink()

    def test_contains_match_validation(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Capital of France?", "answer": "Paris"}\n')
            f.flush()
            path = Path(f.name)

        try:
            ps = JSONLProblemSet.with_contains_match(path)
            problem = list(ps)[0]
            assert ps.validate(problem, "Paris")
            assert ps.validate(problem, "The answer is Paris")
            assert not ps.validate(problem, "London")
        finally:
            path.unlink()

    def test_custom_validator(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "test", "answer": 42}\n')
            f.flush()
            path = Path(f.name)

        try:
            def numeric_validator(problem: Problem, solution: str) -> bool:
                try:
                    return int(solution) == problem.answer
                except ValueError:
                    return False

            ps = JSONLProblemSet(path, numeric_validator)
            problem = list(ps)[0]
            assert ps.validate(problem, "42")
            assert not ps.validate(problem, "not a number")
        finally:
            path.unlink()

    def test_skip_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Q1", "answer": "A1"}\n')
            f.write('\n')  # Empty line
            f.write('{"prompt": "Q2", "answer": "A2"}\n')
            f.flush()
            path = Path(f.name)

        try:
            ps = JSONLProblemSet.with_exact_match(path)
            assert len(ps) == 2
        finally:
            path.unlink()

    def test_with_metadata(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "Q1", "answer": "A1", "metadata": {"difficulty": "easy"}}\n')
            f.flush()
            path = Path(f.name)

        try:
            ps = JSONLProblemSet.with_exact_match(path)
            problem = list(ps)[0]
            assert problem.metadata["difficulty"] == "easy"
        finally:
            path.unlink()
