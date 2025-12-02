"""
CDCL SAT Solver - Implementation following the provided pseudocode exactly
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CDCLResult:
    """Result from CDCL solver"""

    satisfiable: bool
    assignment: Dict[int, bool]
    decisions: int = 0
    backtracks: int = 0
    conflicts: int = 0
    learned_clauses: int = 0


class CDCLSolver:
    """
    CDCL SAT Solver following pseudocode structure:
    - Main CDCL loop
    - Assign helper
    - Propagate (unit propagation)
    - AnalyzeConflict (First-UIP)
    - Backjump
    """

    def __init__(self):
        self.clauses: List[List[int]] = []
        self.variables: int = 0
        self.assignment: Dict[int, bool] = {}
        self.trail: List[int] = []
        self.level: Dict[int, int] = {}
        self.reason: Dict[int, Optional[List[int]]] = {}

    def solve(self, clauses: List[List[int]], variables: int) -> CDCLResult:
        """
        Main CDCL algorithm
        """
        # Initialize
        self.clauses = clauses.copy()
        self.variables = variables
        self.assignment = {}
        self.trail = []
        self.level = {}
        self.reason = {}

        current_level = 0
        decisions = 0
        conflicts = 0
        learned = 0

        # Check for empty clauses
        if any(len(c) == 0 for c in self.clauses):
            return CDCLResult(False, {}, 0, 0, 0, 0)

        # Initial unit propagation at decision level 0
        conflict = self._propagate()
        if conflict is not None:
            return CDCLResult(False, {}, 0, 0, 1, 0)

        # Main loop
        while True:
            # If all variables assigned, return SAT
            if self._all_variables_assigned():
                return CDCLResult(
                    satisfiable=True,
                    assignment=self.assignment.copy(),
                    decisions=decisions,
                    backtracks=conflicts,
                    conflicts=conflicts,
                    learned_clauses=learned,
                )

            # Make decision
            lit = self._choose_decision_literal()
            current_level = current_level + 1
            decisions += 1
            self._assign(lit, current_level, None)

            # Propagate and handle conflicts
            while True:
                conflict = self._propagate()

                if conflict is None:
                    # No conflict, continue to next decision
                    break

                # Conflict at decision level 0 means UNSAT
                if current_level == 0:
                    return CDCLResult(
                        False,
                        {},
                        decisions,
                        conflicts,
                        conflicts,
                        learned,
                    )

                # Analyze conflict and learn clause
                learned_clause, backjump_level = self._analyze_conflict(conflict)

                # Add learned clause
                self.clauses.append(learned_clause)
                learned += 1
                conflicts += 1

                # Backjump
                self._backjump(backjump_level)
                current_level = backjump_level
            # end while
        # end while

    #  Helper

    def _assign(self, lit: int, lvl: int, reason_clause: Optional[List[int]]):
        """
        Assign function from pseudocode
        """
        var = abs(lit)
        value = lit > 0
        self.assignment[var] = value
        self.level[var] = lvl
        self.reason[var] = reason_clause
        self.trail.append(var)

    def _propagate(self) -> Optional[List[int]]:
        """
        Propagate function - simple version from pseudocode
        Returns conflict clause if conflict, None otherwise
        """
        changed = True

        while changed:
            changed = False

            for clause in self.clauses:
                true_count = 0
                false_count = 0
                unassigned_lit = None

                # Evaluate each literal in clause
                for lit in clause:
                    var = abs(lit)

                    if var in self.assignment:
                        # Check if literal is true
                        if (lit > 0 and self.assignment[var]) or (
                            lit < 0 and not self.assignment[var]
                        ):
                            true_count += 1
                        else:
                            false_count += 1
                    else:
                        unassigned_lit = lit

                # Clause already satisfied
                if true_count > 0:
                    continue

                # Conflict: all literals are false
                if false_count == len(clause):
                    return clause

                # Unit clause: exactly one unassigned literal
                if false_count == len(clause) - 1 and unassigned_lit is not None:
                    # Get current level from last variable in trail, or 0
                    current_level = self.level[self.trail[-1]] if self.trail else 0
                    self._assign(unassigned_lit, current_level, clause)
                    changed = True

        return None  # No conflict

    def _analyze_conflict(self, conflict: List[int]) -> Tuple[List[int], int]:
        """
        AnalyzeConflict function - First-UIP from pseudocode
        Returns (learned_clause, backjump_level)
        """
        if not self.trail or self.level[self.trail[-1]] == 0:
            return ([], -1)

        current_level = self.level[self.trail[-1]]

        # Start with conflict clause
        learned = conflict.copy()

        # Get variables in conflict that are at current level
        seen = {abs(lit) for lit in conflict}
        counter = 0

        for lit in learned:
            var = abs(lit)
            if self.level.get(var, 0) == current_level:
                counter += 1

        # Path index - start from end of trail
        path_index = len(self.trail) - 1

        # Resolve until First-UIP (one literal at current level)
        while counter > 1:
            # Find next literal on trail at current level in learned clause
            while path_index >= 0:
                var = self.trail[path_index]
                if var in seen and self.level.get(var, 0) == current_level:
                    break
                path_index -= 1

            if path_index < 0:
                break

            # Get variable and its reason
            p = self.trail[path_index]
            reason_clause = self.reason.get(p)
            path_index -= 1

            # If no reason (decision variable), skip
            if reason_clause is None:
                counter -= 1
                continue

            # Resolve: remove p from learned, add other literals from reason
            learned = [lit for lit in learned if abs(lit) != p]

            for lit in reason_clause:
                var = abs(lit)
                if var not in seen:
                    seen.add(var)
                    learned.append(lit)
                    if self.level.get(var, 0) == current_level:
                        counter += 1

            counter -= 1
        # end while

        # Compute backjump level - second highest level in learned clause
        backjump_level = 0
        levels = []
        for lit in learned:
            var = abs(lit)
            lvl = self.level.get(var, 0)
            if lvl < current_level:
                levels.append(lvl)

        if levels:
            backjump_level = max(levels)

        return (learned, backjump_level)

    def _backjump(self, target_level: int):
        """
        Backjump function from pseudocode
        """
        while self.trail and self.level.get(self.trail[-1], 0) > target_level:
            var = self.trail.pop()
            del self.assignment[var]
            if var in self.level:
                del self.level[var]
            if var in self.reason:
                del self.reason[var]

    def _all_variables_assigned(self) -> bool:
        """Check if all variables are assigned"""
        return len(self.assignment) == self.variables

    def _choose_decision_literal(self) -> int:
        """Choose next decision literal (simple: first unassigned variable)"""
        for var in range(1, self.variables + 1):
            if var not in self.assignment:
                return var  # Default to positive literal
        return 1  # Fallback
