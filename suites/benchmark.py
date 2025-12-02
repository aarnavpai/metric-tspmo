"""
Unified suite for testing and benchmarking solvers.

This module provides the test function that is parametrized by pytest to run
every solver against every compatible problem. It measures execution time,
memory consumption, and validates correctness.
"""

import time
import json
import csv
import pathlib
from typing import Tuple, Any

import pytest

from parser import Problem
from .conftest import SOLVER_REGISTRY, PROBLEM_REGISTRY


@pytest.mark.timeout(120)
def test_solver_correctness(
    solver_name: str, problem_info: Tuple[str, str, Any]
) -> None:
    """
    Execute a single benchmark test run and write results to disk.

    Args:
        solver_name: The name key for the solver to instantiate.
        problem_info: A tuple containing (type, name, path) of the problem.
    """
    problem_type, problem_name, problem_path = problem_info

    # retrieve the adapter class capable of parsing this problem type.
    problem_class = PROBLEM_REGISTRY[problem_type]
    problem_adapter = problem_class()

    # read and parse the problem file.
    with open(problem_path, "r") as f:
        problem: Problem = problem_adapter.load(f.read())

    # retrieve the solver class. the registry stores a tuple of
    # (class, supported_types), so we grab index 0.
    solver_class = SOLVER_REGISTRY[solver_name][0]
    solver = solver_class()

    # capture start times. perf_counter is best for measuring duration (wall)
    # while process_time measures cpu cycles consumed by the thread.
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    # run the solver on the given problem.
    result = solver.solve(problem)

    # calculate elapsed time immediately after the solver returns.
    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    # ensure the solver returned a boolean, not None or some other type.
    assert isinstance(result.satisfiable, bool)

    verified = False
    if result.satisfiable and result.assignment is not None:
        # if the solver claims satisfiable, we must verify the assignment.
        # the verify method checks if the assignment satisfies all clauses.
        verified = solver.verify(problem, result.assignment)
        assert verified, "solver returned invalid assignment"

    # construct a unique identifier for this run. this matches the ID
    # format used in conftest.py's parametrization.
    safe_id = f"{solver_name}:{problem_type}:{problem_name}"

    # define the output path. we write to individual files to avoid file
    # locking issues when multiple tests run in parallel on different cores.
    project_root = pathlib.Path(__file__).parent.parent
    output_dir = project_root / "result" / "benchs"
    output_file = output_dir / f"{safe_id}.csv"

    # define csv headers. we include an 'id' column to facilitate easy
    # joining with memray data in the reporting phase.
    headers = [
        "id",
        "solver",
        "problem",
        "type",
        "status",
        "verified",
        "wall_time_sec",
        "cpu_time_sec",
        "stats",
    ]

    # serialize solver-specific statistics (like decisions, backtracks)
    # into a single json string so they fit in one csv column.
    stats_json = json.dumps(result.stats)
    status_str = "sat" if result.satisfiable else "unsat"

    # write the file. using 'w' mode overwrites any previous run of this
    # specific test case, ensuring we don't accumulate duplicate rows.
    # newline="" is required by the csv module to handle line endings correctly.
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(
            [
                safe_id,
                solver_name,
                problem_name,
                problem_type,
                status_str,
                verified,
                f"{wall_elapsed:.6f}",
                f"{cpu_elapsed:.6f}",
                stats_json,
            ]
        )
