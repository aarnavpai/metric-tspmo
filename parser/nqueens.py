from typing import List, Dict
from .problem import Problem
from .cnf import Cnf, Clause, Literal


class NQueens(Problem):
    """
    Represents the N-Queens problem.
    """

    def __init__(self, n: int) -> None:
        """
        Initialize N-Queens problem.

        Args:
            n: Board size (NxN) and number of queens.
        """
        self.n = n

    def __repr__(self) -> str:
        return f"NQueens(n={self.n})"

    @classmethod
    def parse(cls, content: str) -> "NQueens":
        """
        Parse N-Queens problem from string.
        Expected format: just the integer N.
        """
        try:
            n = int(content.strip())
            return cls(n)
        except ValueError:
            raise ValueError(f"Invalid N-Queens input: {content}")

    def cnf(self) -> Cnf:
        """
        Encode N-Queens problem to CNF.
        """
        n = self.n
        clauses: List[Clause] = []

        def var(r: int, c: int) -> int:
            # Map (row r, col c) to a DIMACS variable number.
            # r, c are 0-indexed; DIMACS variables start at 1.
            return r * n + c + 1

        def at_most_one(lits: List[int]) -> List[Clause]:
            """
            Encode AT MOST ONE using pairwise negation.
            """
            new_clauses = []
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    # ~xi OR ~xj
                    new_clauses.append(
                        Clause([Literal(lits[i], True), Literal(lits[j], True)])
                    )
            return new_clauses

        def exactly_one(lits: List[int]) -> List[Clause]:
            """
            Encode EXACTLY ONE.
            """
            new_clauses = []
            # at least one: (x1 OR x2 OR ... OR xn)
            new_clauses.append(Clause([Literal(x, False) for x in lits]))

            # at most one
            new_clauses.extend(at_most_one(lits))
            return new_clauses

        # 1. Row constraints: exactly 1 queen in each row
        for r in range(n):
            lits = [var(r, c) for c in range(n)]
            clauses.extend(exactly_one(lits))

        # 2. Column constraints: at most 1 queen in each column
        for c in range(n):
            lits = [var(r, c) for r in range(n)]
            clauses.extend(at_most_one(lits))

        # 3. Diagonal constraints (main diagonals r - c)
        diags: Dict[int, List[int]] = {}
        for r in range(n):
            for c in range(n):
                d = r - c
                diags.setdefault(d, []).append(var(r, c))

        for d in diags:
            clauses.extend(at_most_one(diags[d]))

        # 4. Anti-diagonal constraints (r + c)
        antidiags: Dict[int, List[int]] = {}
        for r in range(n):
            for c in range(n):
                d = r + c
                antidiags.setdefault(d, []).append(var(r, c))

        for d in antidiags:
            clauses.extend(at_most_one(antidiags[d]))

        return Cnf(set(range(1, n * n + 1)), clauses)
