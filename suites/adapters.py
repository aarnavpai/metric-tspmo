"""
Adapters for solvers and problems to provide a unified testing interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

from parser import Cnf, Sudoku, Problem, NQueens, Clique, HamPath
from solver import (
    DPLLSolver,
    CDCLSolver,
    SudokuSolver,
    SchöningSolver,
    BacktrackingSolver,
)

from .conftest import register_solver, register_problem


class ProblemAdapter(ABC):
    """
    Base class for adapters that load problem files.

    These adapters are responsible for parsing raw file content into
    structured objects that implement the Problem interface.
    """

    @abstractmethod
    def load(self, content: str) -> Problem:
        """
        Parse file content into a domain object inheriting from Problem.

        Args:
            content: Raw text content of the problem file.

        Returns:
            An instance of Cnf, Sudoku, or other Problem subclass.
        """
        pass


@register_problem("cnf", ".cnf")
class CnfProblemAdapter(ProblemAdapter):
    """Adapter for loading DIMACS CNF files."""

    def load(self, content: str) -> Cnf:
        """Parse content and return a Cnf object."""
        return Cnf.parse(content)


@register_problem("sudoku", ".sdku")
class SudokuProblemAdapter(ProblemAdapter):
    """Adapter for loading Sudoku text files."""

    def load(self, content: str) -> Sudoku:
        """Parse content and return a Sudoku object."""
        return Sudoku.parse(content)


@register_problem("nqueens", ".nqueens")
class NQueensProblemAdapter(ProblemAdapter):
    """Adapter for loading N-Queens problem files."""

    def load(self, content: str) -> NQueens:
        """Parse content and return a NQueens object."""
        return NQueens.parse(content)


@register_problem("clique", ".graph")
class CliqueProblemAdapter(ProblemAdapter):
    """Adapter for loading K-Clique problem files."""

    def load(self, content: str) -> Clique:
        """Parse content and return a K-Clique object."""
        return Clique.parse(content)


@register_problem("hampath", ".graph")
class HamPathProblemAdapter(ProblemAdapter):
    """Adapter for loading Hamiltonian Path problem files."""

    def load(self, content: str) -> HamPath:
        """Parse content and return a Hamiltonian Path object."""
        return HamPath.parse(content)


@dataclass
class SolverResult:
    """Unified result format for all solvers."""

    satisfiable: bool
    assignment: Optional[Dict[int, bool]] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class SolverAdapter(ABC):
    """Base class for solver adapters."""

    @abstractmethod
    def solve(self, problem: Problem) -> SolverResult:
        """
        Solve the given Problem instance.

        Adapters must handle any necessary conversion (e.g., to CNF)
        before passing the data to the underlying solver implementation.
        """
        pass

    @abstractmethod
    def verify(self, problem: Problem, assignment: Dict[int, bool]) -> bool:
        """Verify the assignment against the problem constraints."""
        pass


@register_solver("dpll", ["cnf", "sudoku", "nqueens", "clique", "hampath"])
class DPLLAdapter(SolverAdapter):
    """Adapter for the DPLL SAT solver."""

    def solve(self, problem: Problem) -> SolverResult:
        """
        Solve any Problem by converting it to CNF first.

        Since the Problem class has a .cnf() method, this adapter can handle
        both native Cnf objects and Sudoku objects automatically.
        """
        cnf_problem = problem.cnf()

        solver = DPLLSolver(cnf_problem)
        result = solver.solve()

        return SolverResult(
            satisfiable=result.satisfiable,
            assignment=result.assignment,
            stats={
                "decisions": result.decisions,
                "unit_propagations": result.unit_propagations,
                "pure_literals": result.pure_literals,
            },
        )

    def verify(self, problem: Problem, assignment: Dict[int, bool]) -> bool:
        """Verify assignment against CNF constraints."""
        # verification is performed on the cnf representation to ensure
        # all logical constraints are strictly met.
        cnf_problem = problem.cnf()
        solver = DPLLSolver(cnf_problem)
        return solver.verify(assignment)


@register_solver("cdcl", ["cnf", "sudoku", "nqueens", "clique", "hampath"])
class CDCLAdapter(SolverAdapter):
    """Adapter for the CDCL SAT solver."""

    def solve(self, problem: Problem) -> SolverResult:
        """
        Solve any Problem by converting it to CNF first.

        This adapter handles the conversion from the object-oriented CNF
        structure to the list-of-lists integer format required by the
        CDCL implementation backend.
        """
        cnf_problem = problem.cnf()

        # convert the object structure to simple integer lists
        clauses: List[List[int]] = []
        for clause in cnf_problem.clauses:
            c = []
            for lit in clause.literals:
                # encode negation as negative integers
                c.append(-lit.id if lit.negated else lit.id)
            clauses.append(c)

        solver = CDCLSolver()
        result = solver.solve(clauses, len(cnf_problem.variables))

        return SolverResult(
            satisfiable=result.satisfiable,
            assignment=result.assignment,
            stats={
                "decisions": result.decisions,
                "conflicts": result.conflicts,
                "learned_clauses": result.learned_clauses,
                "backtracks": result.backtracks,
            },
        )

    def verify(self, problem: Problem, assignment: Dict[int, bool]) -> bool:
        """Verify assignment using DPLL logic on CNF representation."""
        # cdcl doesn't have a built-in verify, so we reuse dpll's verify.
        cnf_problem = problem.cnf()
        solver = DPLLSolver(cnf_problem)
        return solver.verify(assignment)


@register_solver("sudoku", ["sudoku"])
class SudokuSolverAdapter(SolverAdapter):
    """Adapter for the specialized Sudoku solver."""

    def solve(self, problem: Problem) -> SolverResult:
        """
        Solve using the specialized Sudoku algorithm.

        This adapter requires the actual Sudoku object. We perform a check
        to ensure the generic Problem passed in is actually a Sudoku instance.
        """
        if not isinstance(problem, Sudoku):
            # checking type safety at runtime since the signature is generic
            raise ValueError("SudokuSolver requires a Sudoku problem instance")

        solver = SudokuSolver(problem)
        result = solver.solve()

        if result is None:
            return SolverResult(satisfiable=False)

        # safeguard against inconsistent state in the underlying solver
        try:
            assignment = result.assignment
        except ValueError:
            return SolverResult(satisfiable=False)

        return SolverResult(satisfiable=True, assignment=assignment, stats={})

    def verify(self, problem: Problem, assignment: Dict[int, bool]) -> bool:
        """
        Verify using CNF constraints.

        We use the generic .cnf() method for verification to ensure
        mathematical correctness of the constraints, providing a ground-truth
        check against the logic solver's result.
        """
        cnf = problem.cnf()
        solver = DPLLSolver(cnf)
        return solver.verify(assignment)


@register_solver("schöning", ["cnf", "sudoku", "nqueens", "clique", "hampath"])
class SchöningAdapter(SolverAdapter):
    """Adapter for the Schöning SAT solver."""

    def solve(self, problem: Problem) -> SolverResult:
        """
        Solve any Problem by converting it to CNF first.
        """
        cnf_problem = problem.cnf()

        clauses: List[List[int]] = []
        for clause in cnf_problem.clauses:
            c = []
            for lit in clause.literals:
                c.append(-lit.id if lit.negated else lit.id)
            clauses.append(c)

        solver = SchöningSolver()
        result = solver.solve(clauses, len(cnf_problem.variables))

        return SolverResult(
            satisfiable=result.satisfiable,
            assignment=result.assignment,
            stats={
                "steps": result.steps,
            },
        )

    def verify(self, problem: Problem, assignment: Dict[int, bool]) -> bool:
        """Verify assignment using DPLL logic on CNF representation."""
        cnf_problem = problem.cnf()
        solver = DPLLSolver(cnf_problem)
        return solver.verify(assignment)


@register_solver("backtracking", ["cnf", "sudoku", "nqueens", "clique"])
class BacktrackingAdapter(SolverAdapter):
    """Adapter for the Naive Backtracking SAT solver."""

    def solve(self, problem: Problem) -> SolverResult:
        """
        Solve any Problem by converting it to CNF first.
        """
        cnf_problem = problem.cnf()

        clauses: List[List[int]] = []
        for clause in cnf_problem.clauses:
            c = []
            for lit in clause.literals:
                c.append(-lit.id if lit.negated else lit.id)
            clauses.append(c)

        solver = BacktrackingSolver()
        result = solver.solve(clauses, len(cnf_problem.variables))

        return SolverResult(
            satisfiable=result.satisfiable,
            assignment=result.assignment,
            stats={
                "recursions": result.recursions,
            },
        )

    def verify(self, problem: Problem, assignment: Dict[int, bool]) -> bool:
        """Verify assignment using DPLL logic on CNF representation."""
        cnf_problem = problem.cnf()
        solver = DPLLSolver(cnf_problem)
        return solver.verify(assignment)
