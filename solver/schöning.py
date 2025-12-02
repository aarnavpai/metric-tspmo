"""
Schöning's Probabilistic solver for k-SAT.

It has a probability of success of at least p = (k/2(k-1))^n,
and a time complexity of O(1/p^n). The implementation follows
the pseudocode provided in the writeup.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
from parser import Cnf


@dataclass
class SchöningResult:
    """Result from Schöning solver"""

    satisfiable: bool
    assignment: Dict[int, bool]
    steps: int = 0


class SchöningSolver:
    """
    Schöning solver following pseudocode structure:
    - Initial random assignment
    - repeat 3|V| times:
      - if satisfied, return
      - else flip a random variable of a random clause
    """

    def __init__(self, cnf: Optional[Cnf] = None):
        if cnf:
            self.variables = len(cnf.variables)
            self.clauses = []
            for clause in cnf.clauses:
                c = []
                for lit in clause.literals:
                    c.append(-lit.id if lit.negated else lit.id)
                self.clauses.append(c)
        else:
            self.clauses: List[List[int]] = []
            self.variables: int = 0

        self.assignment: Dict[int, bool] = {}

    def solve(
        self,
        clauses: Optional[List[List[int]]] = None,
        variables: Optional[int] = None,
        restarts: int = 300,
    ) -> SchöningResult:
        """
        Main Schöning algorithm with optimized incremental updates.

        The number of restarts can be adjusted. The theoretical bound
        suggests t = n * (2(k-1)/k)^n, which can be large.
        """
        if clauses is not None:
            self.clauses = clauses
        if variables is not None:
            self.variables = variables

        # Check for empty clauses
        if not self.clauses:
            return SchöningResult(True, {}, 0)

        if any(len(c) == 0 for c in self.clauses):
            return SchöningResult(False, {}, 0)

        # Pre-process: build adjacency list
        # var_to_clauses[v] = [(clause_idx, literal), ...]
        var_to_clauses: List[List[Tuple[int, int]]] = [
            [] for _ in range(self.variables + 1)
        ]
        for c_idx, clause in enumerate(self.clauses):
            for lit in clause:
                var_to_clauses[abs(lit)].append((c_idx, lit))

        total_steps = 0

        for _ in range(restarts):
            # Random Assignment
            # 1-based indexing for variables
            assignment = [False] * (self.variables + 1)
            for i in range(1, self.variables + 1):
                assignment[i] = random.choice([True, False])

            # Initialize sat counts and unsatisfied list
            clause_sat_counts = [0] * len(self.clauses)
            unsatisfied_indices = []

            for c_idx, clause in enumerate(self.clauses):
                sat_count = 0
                for lit in clause:
                    var = abs(lit)
                    val = assignment[var]
                    # lit is satisfied if (lit > 0 and val) or (lit < 0 and not val)
                    # equivalent to: (lit > 0) == val
                    if (lit > 0) == val:
                        sat_count += 1
                clause_sat_counts[c_idx] = sat_count
                if sat_count == 0:
                    unsatisfied_indices.append(c_idx)

            # Map for O(1) removal: clause_idx -> position in unsatisfied_indices
            clause_pos_in_unsat = [-1] * len(self.clauses)
            for i, c_idx in enumerate(unsatisfied_indices):
                clause_pos_in_unsat[c_idx] = i

            steps = 0
            max_steps = 3 * self.variables

            for _ in range(max_steps):
                if not unsatisfied_indices:
                    final_assignment = {
                        i: assignment[i] for i in range(1, self.variables + 1)
                    }
                    self.assignment = final_assignment
                    return SchöningResult(True, final_assignment, total_steps + steps)

                # Pick random unsatisfied clause
                rand_idx = random.randrange(len(unsatisfied_indices))
                c_idx = unsatisfied_indices[rand_idx]
                clause = self.clauses[c_idx]

                # Pick random literal to flip
                lit_to_flip = random.choice(clause)
                var_to_flip = abs(lit_to_flip)

                # Flip variable
                assignment[var_to_flip] = not assignment[var_to_flip]
                new_val = assignment[var_to_flip]

                # Update affected clauses
                for affected_c_idx, affected_lit in var_to_clauses[var_to_flip]:
                    # Check if this literal now satisfies (is_now_true)
                    is_now_true = (affected_lit > 0) == new_val

                    if is_now_true:
                        clause_sat_counts[affected_c_idx] += 1
                        if clause_sat_counts[affected_c_idx] == 1:
                            # Became satisfied, remove from unsat list
                            pos = clause_pos_in_unsat[affected_c_idx]

                            # Swap with last
                            last_idx = len(unsatisfied_indices) - 1
                            last_c_idx = unsatisfied_indices[last_idx]

                            unsatisfied_indices[pos] = last_c_idx
                            clause_pos_in_unsat[last_c_idx] = pos

                            unsatisfied_indices.pop()
                            clause_pos_in_unsat[affected_c_idx] = -1
                    else:
                        clause_sat_counts[affected_c_idx] -= 1
                        if clause_sat_counts[affected_c_idx] == 0:
                            # Became unsatisfied, add to unsat list
                            unsatisfied_indices.append(affected_c_idx)
                            clause_pos_in_unsat[affected_c_idx] = (
                                len(unsatisfied_indices) - 1
                            )

                steps += 1

            total_steps += steps

        return SchöningResult(False, {}, total_steps)
