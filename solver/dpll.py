"""
DPLL Algorithm Implementation

The DPLL algorithm is a complete, backtracking-based search algorithm for deciding
the satisfiability of propositional logic formulae in conjunctive normal form (CNF).

Algorithm Overview:
1. Base Cases:
   - If formula is empty (no clauses), return SAT
   - If formula contains an empty clause, return UNSAT

2. Unit Propagation:
   - Find unit clauses (clauses with only one literal)
   - Assign the literal's value to satisfy the clause
   - Simplify formula based on assignment

3. Pure Literal Elimination:
   - Find literals that appear with only one polarity
   - Assign values to satisfy all occurrences
   - Simplify formula

4. Branching (DPLL-Decide):
   - Choose an unassigned variable
   - Try assigning it True, recursively solve
   - If fails, try assigning it False, recursively solve
   - Backtrack if both fail
"""

from typing import Optional, Dict, Set, List, Tuple
from dataclasses import dataclass
from parser import Cnf, Clause


@dataclass
class DPLLResult:
    """Result of DPLL algorithm execution"""

    satisfiable: bool
    assignment: Optional[Dict[int, bool]] = None  # variable_id -> boolean value
    decisions: int = 0
    unit_propagations: int = 0
    pure_literals: int = 0


class DPLLSolver:
    def __init__(self, cnf: Cnf):
        """
        Initialize DPLL solver with a CNF formula

        """
        self.original_cnf = cnf
        self.variables = cnf.variables.copy()

        self.decisions = 0
        self.unit_propagations = 0
        self.pure_literals = 0

    def solve(self) -> DPLLResult:
        """
        Solve the SAT problem using DPLL algorithm

        Returns:
            DPLLResult containing satisfiability and assignment (if SAT)
        """
        # reset statistics
        self.decisions = 0
        self.unit_propagations = 0
        self.pure_literals = 0

        # convert to internal representation
        clauses = self._convert_to_internal(self.original_cnf.clauses)
        assignment = {}

        # run DPLL
        result = self._dpll(clauses, assignment)

        return DPLLResult(
            satisfiable=result is not None,
            assignment=result if result is not None else None,
            decisions=self.decisions,
            unit_propagations=self.unit_propagations,
            pure_literals=self.pure_literals,
        )

    def _convert_to_internal(self, clauses: List[Clause]) -> List[Set[int]]:
        """
        Convert parser's Clause representation to internal representation

        The internal representation is a set of integers where:
        - positive integer n represents variable n
        - negative integer -n represents NOT variable n

        Args:
            clauses: List of Clause objects from parser

        Returns:
            List of sets, each set representing a clause
        """
        internal_clauses = []
        for clause in clauses:
            internal_clause = set()
            for literal in clause.literals:
                if literal.negated:
                    internal_clause.add(-literal.id)
                else:
                    internal_clause.add(literal.id)
            internal_clauses.append(internal_clause)
        return internal_clauses

    def _dpll(
        self, clauses: List[Set[int]], assignment: Dict[int, bool]
    ) -> Optional[Dict[int, bool]]:
        """
        Core DPLL algorithm implementation

        Returns:
            Complete satisfying assignment if SAT, None if UNSAT
        """
        # base case 1: empty formula (all clauses satisfied)
        if not clauses:
            return assignment

        # base case 2: empty clause exists (conflict)
        if any(len(clause) == 0 for clause in clauses):
            return None

        # unit propagation
        unit_result = self._unit_propagate(clauses, assignment)
        if unit_result is None:
            return None
        clauses, assignment = unit_result

        # check if formula is satisfied after unit propagation
        if not clauses:
            return assignment

        # check for empty clause after unit propagation
        if any(len(clause) == 0 for clause in clauses):
            return None

        # pure literal elimination
        pure_result = self._pure_literal_eliminate(clauses, assignment)
        if pure_result is not None:
            clauses, assignment = pure_result

            # check if formula is satisfied after pure literal elimination
            if not clauses:
                return assignment

        # branching (DPLL-decide)
        self.decisions += 1

        # choose an unassigned variable
        variable = self._choose_variable(clauses, assignment)

        # try assigning variable = True
        new_assignment = assignment.copy()
        new_assignment[variable] = True
        new_clauses = self._simplify(clauses, variable, True)

        result = self._dpll(new_clauses, new_assignment)
        if result is not None:
            return result

        # backtrack: try assigning variable = False
        new_assignment = assignment.copy()
        new_assignment[variable] = False
        new_clauses = self._simplify(clauses, variable, False)

        return self._dpll(new_clauses, new_assignment)

    def _unit_propagate(
        self, clauses: List[Set[int]], assignment: Dict[int, bool]
    ) -> Optional[Tuple[List[Set[int]], Dict[int, bool]]]:
        assignment = assignment.copy()

        while True:
            unit_clauses = [clause for clause in clauses if len(clause) == 1]

            if not unit_clauses:
                break

            # process all unit clauses
            for unit_clause in unit_clauses:
                literal = next(iter(unit_clause))  # get the single literal
                variable = abs(literal)
                value = literal > 0

                # check for conflict
                if variable in assignment and assignment[variable] != value:
                    return None

                # make assignment
                assignment[variable] = value
                self.unit_propagations += 1

                # simplify clauses
                clauses = self._simplify(clauses, variable, value)

                # check for empty clause
                if any(len(clause) == 0 for clause in clauses):
                    return None

        return clauses, assignment

    def _pure_literal_eliminate(
        self, clauses: List[Set[int]], assignment: Dict[int, bool]
    ) -> Optional[Tuple[List[Set[int]], Dict[int, bool]]]:
        # find all literals and their polarities
        positive_vars = set()
        negative_vars = set()

        for clause in clauses:
            for literal in clause:
                if literal > 0:
                    positive_vars.add(literal)
                else:
                    negative_vars.add(-literal)

        # find pure literals - only one polarity
        pure_positive = positive_vars - negative_vars
        pure_negative = negative_vars - positive_vars

        if not pure_positive and not pure_negative:
            return None

        assignment = assignment.copy()

        # assign pure literals
        for var in pure_positive:
            if var not in assignment:
                assignment[var] = True
                self.pure_literals += 1
                clauses = self._simplify(clauses, var, True)

        for var in pure_negative:
            if var not in assignment:
                assignment[var] = False
                self.pure_literals += 1
                clauses = self._simplify(clauses, var, False)

        return clauses, assignment

    def _simplify(
        self, clauses: List[Set[int]], variable: int, value: bool
    ) -> List[Set[int]]:
        literal = variable if value else -variable
        opposite_literal = -literal

        simplified = []
        for clause in clauses:
            if literal in clause:
                # clause is satisfied, remove it
                continue
            elif opposite_literal in clause:
                # remove the false literal from the clause
                new_clause = clause - {opposite_literal}
                simplified.append(new_clause)
            else:
                # clause is unaffected
                simplified.append(clause.copy())

        return simplified

    def _choose_variable(
        self, clauses: List[Set[int]], assignment: Dict[int, bool]
    ) -> int:
        # count occurrences of each variable
        var_count = {}
        for clause in clauses:
            for literal in clause:
                var = abs(literal)
                if var not in assignment:
                    var_count[var] = var_count.get(var, 0) + 1

        # choose variable with highest count
        if var_count:
            return max(var_count, key=var_count.get)

        # fallback: choose any unassigned variable
        for var in self.variables:
            if var not in assignment:
                return var

        # should not reach here
        return min(self.variables)

    def verify(self, assignment: Dict[int, bool]) -> bool:
        """
        Verify if an assignment satisfies the original formula.
        """
        for clause in self.original_cnf.clauses:
            clause_satisfied = False
            for literal in clause.literals:
                var_value = assignment.get(literal.id, None)
                if var_value is None:
                    continue

                literal_value = var_value if not literal.negated else not var_value
                if literal_value:
                    clause_satisfied = True
                    break

            if not clause_satisfied:
                return False

        return True
