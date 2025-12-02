"""
Naive Backtracking Solver for CNF.

This implementation uses a simple recursive backtracking algorithm (O(2^n))
to find a satisfying assignment. It serves as a baseline for performance comparisons.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BacktrackingResult:
    """Result from Backtracking solver"""

    satisfiable: bool
    assignment: Dict[int, bool]
    recursions: int = 0


class BacktrackingSolver:
    """
    Naive backtracking solver.
    """

    def __init__(self):
        self.clauses: List[List[int]] = []
        self.assignment: Dict[int, bool] = {}
        self.recursions: int = 0

    def solve(self, clauses: List[List[int]], variables: int) -> BacktrackingResult:
        """
        Main Backtracking algorithm
        """
        self.clauses = clauses
        self.assignment = {}
        self.recursions = 0

        vars = list(range(1, variables + 1))

        if self._backtrack(vars, 0):
            return BacktrackingResult(True, self.assignment.copy(), self.recursions)

        return BacktrackingResult(False, {}, self.recursions)

    def _backtrack(self, variables: List[int], index: int) -> bool:
        self.recursions += 1

        # Check consistency of current partial assignment
        if not self._is_consistent():
            return False

        # Base case: all variables assigned and consistent
        if index == len(variables):
            return True

        var = variables[index]

        # Try True
        self.assignment[var] = True
        if self._backtrack(variables, index + 1):
            return True

        # Try False
        self.assignment[var] = False
        if self._backtrack(variables, index + 1):
            return True

        # Backtrack
        del self.assignment[var]
        return False

    def _is_consistent(self) -> bool:
        """
        Check if the current partial assignment is consistent with all clauses.
        A clause is inconsistent if all its literals are assigned and evaluate to False.
        """
        for clause in self.clauses:
            clause_satisfied = False
            all_assigned = True

            for lit in clause:
                var = abs(lit)
                val = self.assignment.get(var)

                if val is None:
                    all_assigned = False
                    continue

                # If literal is true under assignment, clause is satisfied
                if (lit > 0 and val) or (lit < 0 and not val):
                    clause_satisfied = True
                    break

            # If clause is already satisfied, it's consistent
            if clause_satisfied:
                continue

            # If not satisfied and all literals are assigned, it's FALSE (inconsistent)
            if all_assigned:
                return False

        return True
