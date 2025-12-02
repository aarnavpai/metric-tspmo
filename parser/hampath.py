from typing import List, Tuple
from collections import defaultdict

from .problem import Problem
from .cnf import Cnf, Clause, Literal


class HamPath(Problem):
    """
    Represents the Hamiltonian Path problem.
    """

    def __init__(self, vertices: int, edges: List[Tuple[int, int]]) -> None:
        self.vertices = vertices
        self.edges = edges

    def __repr__(self) -> str:
        return f"HamPath(n={self.vertices}, edges={self.edges})"

    @classmethod
    def parse(cls, content: str) -> "HamPath":
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            raise ValueError("hampath: empty file")

        vertices = int(lines[0])
        edges = []
        for line in lines[1:]:
            points = line.split()
            if len(points) != 2:
                raise ValueError("hampath: each edge must have two end points")
            # Convert 0-based input to 1-based internal representation
            edges.append((int(points[0]) + 1, int(points[1]) + 1))

        return cls(vertices, edges)

    def cnf(self) -> Cnf:
        n = self.vertices
        clauses = []

        # we map (i, j) to a linear integer 1..n^2 -> (i*n)+j+1
        def var(idx, pos):
            return ((idx - 1) * n) + pos

        # Build adjacency list (undirected)
        adj = defaultdict(set)
        for u, v in self.edges:
            adj[u].add(v)
            adj[v].add(u)

        # 1. Each vertex must appear at least once in the path.
        for i in range(1, n + 1):
            clauses.append(Clause([Literal(var(i, j), False) for j in range(1, n + 1)]))

        # 2. Each vertex must appear at most once in the path.
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for k in range(j + 1, n + 1):
                    clauses.append(
                        Clause([Literal(var(i, j), True), Literal(var(i, k), True)])
                    )

        # 3. Each position must be occupied by at least one vertex.
        for j in range(1, n + 1):
            clauses.append(Clause([Literal(var(i, j), False) for i in range(1, n + 1)]))

        # 4. Each position must be occupied by at most one vertex.
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                for k in range(i + 1, n + 1):
                    clauses.append(
                        Clause([Literal(var(i, j), True), Literal(var(k, j), True)])
                    )

        # 5. Edge constraints (Transition constraints)
        # If vertex u is at position j, then the vertex at position j+1 MUST be a neighbor of u.
        for j in range(1, n):  # positions 1 to n-1
            for u in range(1, n + 1):
                # Literals: ~var(u, j)
                lits = [Literal(var(u, j), True)]
                # Add neighbors at j+1
                for v in adj[u]:
                    lits.append(Literal(var(v, j + 1), False))

                clauses.append(Clause(lits))

        return Cnf(set(range(1, (n * n) + 1)), clauses)
