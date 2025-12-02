from typing import List, Tuple

from .problem import Problem
from .cnf import Cnf, Clause, Literal


class Clique(Problem):
    """
    Represents the k-clique problem.
    """

    def __init__(self, vertices: int, edges: List[Tuple[int, int]]) -> None:
        self.vertices = vertices
        self.edges = edges

    def __repr__(self) -> str:
        return f"Clique(n={self.vertices}, edges={self.edges})"

    @classmethod
    def parse(cls, content: str) -> "Clique":
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("clique: empty file")

        vertices = int(lines[0])
        edges = []
        for line in lines[1:]:
            points = line.split()
            if len(points) != 2:
                raise ValueError("clique: each edge must have two end points")
            # Convert 0-based input to 1-based internal representation
            edges.append((int(points[0]) + 1, int(points[1]) + 1))

        return cls(vertices, edges)

    def cnf(self) -> Cnf:
        n, clauses = self.vertices, []
        k = n // 2

        if k == 0:
            return Cnf(set(), [])

        base = n + 1

        # define the helper function to determine the variable number for
        # the auxiliary counters.
        def counter(i: int, j: int) -> int:
            return base + ((i - 1) * k) + (j - 1)

        # get the total number of variables and sorted edges.
        t = counter(n, k)
        edges = set()
        for u, v in self.edges:
            edges.add((u, v))
            edges.add((v, u))

        # if two nodes u and v are not connected, they cannot both be in the clique.
        for u in range(1, n + 1):
            for v in range(u + 1, n + 1):
                if (u, v) not in edges:
                    clauses.append(Clause([Literal(u, True), Literal(v, True)]))

        # s[1][1] = true if node 1 is selected
        clauses.append(
            Clause(
                [
                    Literal(1, True),  # Not x_1
                    Literal(counter(1, 1), False),  # or s[1][1]  => (x_1 -> s[1][1])
                ]
            )
        )

        # s[1][j] for j > 1 must be false (cannot have count > 1 at first node)
        for j in range(2, k + 1):
            clauses.append(Clause([Literal(counter(1, j), True)]))

        for i in range(2, n + 1):
            x_i = i

            # x_i -> s[i][1]  and  s[i-1][1] -> s[i][1]
            s_curr_1 = counter(i, 1)
            s_prev_1 = counter(i - 1, 1)

            clauses.append(Clause([Literal(x_i, True), Literal(s_curr_1, False)]))
            clauses.append(Clause([Literal(s_prev_1, True), Literal(s_curr_1, False)]))

            for j in range(2, k + 1):
                s_curr = counter(i, j)
                s_prev_same = counter(i - 1, j)
                s_prev_minus = counter(i - 1, j - 1)

                # if count was already j, it remains j. s[i-1][j] -> s[i][j]
                clauses.append(
                    Clause([Literal(s_prev_same, True), Literal(s_curr, False)])
                )

                # if count was j-1 and we pick x_i, count becomes j.
                # (x_i and s[i-1][j-1]) -> s[i][j]
                clauses.append(
                    Clause(
                        [
                            Literal(x_i, True),
                            Literal(s_prev_minus, True),
                            Literal(s_curr, False),
                        ]
                    )
                )

        # the counter at the last node (n) must have reached at least k
        clauses.append(Clause([Literal(counter(n, k), False)]))

        return Cnf(set(range(1, t + 1)), clauses)
