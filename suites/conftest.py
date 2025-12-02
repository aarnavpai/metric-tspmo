"""
Pytest configuration and shared fixtures for the solver benchmarking suite.

This module provides the base for the testing framework. It handles the
registration of solvers and problem types, discovers and loads test files, and
and aggregates performance metrics (time and memory) into a CSV report.
"""

import pathlib
from typing import Dict, List, Type, Any, Tuple

from _pytest.config import Config
from _pytest.python import Metafunc

# registry to map solver names to their implementation classes and supported
# problem types. keys are strings (e.g., "dpll") and values are tuples
# containing the class and a list of supported file extensions.
SOLVER_REGISTRY: Dict[str, Tuple[Type[Any], List[str]]] = {}


def register_solver(name: str, supported_types: List[str]):
    """
    Decorator to register a solver class with the test runner.

    Args:
        name: The unique identifier for the solver.
        supported_types: A list of problem types (e.g., 'cnf') this solver
            can handle.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        # store the class reference and its capabilities in the global
        # registry. this allows the test generator to lookup the class later
        # without hardcoding imports in the test function itself.
        SOLVER_REGISTRY[name] = (cls, supported_types)
        return cls

    return decorator


# registry to map problem type identifiers to their adapter classes.
PROBLEM_REGISTRY: Dict[str, Type[Any]] = {}


def register_problem(name: str, extension: str):
    """
    Decorator to register a problem adapter class.

    Args:
        name: The name of the problem type.
        extension: The file extension associated with this problem type.
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        # attach the extension to the class object. this makes it easy to
        # glob for files of this specific type during test generation.
        cls.extension = extension
        PROBLEM_REGISTRY[name] = cls
        return cls

    return decorator


def pytest_configure(config: Config) -> None:
    """
    Hook called before the test session starts.

    This is used to set up the environment before any tests are collected or
    run. When using pytest-xdist, this runs in the main process before
    workers are spawned (or in the workers, depending on xdist version, but
    mkdir with exist_ok is atomic enough for our needs).
    """
    # resolve the project root relative to this file's location.
    project_root = pathlib.Path(__file__).parent.parent

    # define the directory where individual csv results will be stored.
    # we separate them into a 'benchs' subdirectory to keep the main
    # result folder clean.
    bench_dir = project_root / "result" / "benchs"

    # create the directory if it doesn't exist. using parents=True ensures
    # the 'result' parent is created too. exist_ok=True prevents errors if
    # the directory already exists (e.g., from a previous run).
    bench_dir.mkdir(parents=True, exist_ok=True)


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """
    Dynamically generate test cases based on available files.

    This hook scans the 'problems' directory and creates a test case for
    every valid combination of registered solver and compatible problem
    file found on the disk.
    """
    # check if the test function requesting fixtures is looking for
    # 'solver_name' and 'problem_info'. if not, we shouldn't parameterize it.
    if (
        "solver_name" in metafunc.fixturenames
        and "problem_info" in metafunc.fixturenames
    ):
        # determine the path to the problems directory relative to this file.
        suites_dir = pathlib.Path(__file__).parent
        problems_dir = suites_dir / "problems"

        all_problems: List[Tuple[str, str, pathlib.Path]] = []

        if problems_dir.exists():
            # iterate through registered problem types (e.g., cnf, sudoku).
            for problem_type, problem_class in PROBLEM_REGISTRY.items():
                extension = problem_class.extension
                problem_subdir = problems_dir / problem_type

                if not problem_subdir.exists():
                    continue

                # collect all files with the matching extension. we sort the
                # list to ensure deterministic test ordering across runs.
                files = sorted(list(problem_subdir.glob(f"*{extension}")))
                for f in files:
                    # store the type, the base name, and the full path for later
                    # use in the test function.
                    all_problems.append((problem_type, f.stem, f))

        test_cases = []
        test_ids = []

        # create the cartesian product of solvers and problems.
        for solver_name, (solver_cls, supported_types) in SOLVER_REGISTRY.items():
            for p_type, p_name, p_path in all_problems:
                # specific exclusion logic: backtracking is too slow for
                # general cnf files, so we skip it unless it is the tiny
                # verification instance.
                if solver_name == "backtracking" and not (
                    p_type == "cnf" and p_name == "00tiny"
                ):
                    continue

                # only generate a test if the solver explicitly supports
                # this problem type.
                if p_type in supported_types:
                    test_cases.append((solver_name, (p_type, p_name, p_path)))

                    # generate a filesystem-safe unique identifier.
                    # this string is used by pytest to name the test node
                    # and by us to name the output csv file.
                    test_ids.append(f"{solver_name}:{p_type}:{p_name}")

        # apply the parametrization. 'ids' argument gives the test cases
        # readable names in the output and is used for matching memray data.
        metafunc.parametrize("solver_name, problem_info", test_cases, ids=test_ids)
