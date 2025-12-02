"""
Parser for DIMACS cnf files.

The original format is described here:
  https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps

Some notes about the format:
- Comment lines (which are supposed to appear above the problem
  line but this parser doesn't care about that) start with a 'c'.
- The problem line appears once in the file and starts with a 'p'.
  Its syntax is: `p cnf <num_variables> <num_clauses>`
  Variables are assumed to be numbered from 1 to <num_variables>
- After the problem line, the clauses appear. Each clause is
  represented by a sequence of numbers, which represent the
  variable, separated by (any, according to this parser)
  whitespace, including newlines.
  If a variable is negated, it appears as a negative number.
  Clauses are terminated by the number 0.
- While not mentioned in the original format, SATLIB cnf files
  have two lines with sole characters `%` and `0` respectively
  at the end of the file, plus a few blank lines, all of which
  are ignored by the parser.
"""

from dataclasses import dataclass

from .problem import Problem


@dataclass
class Literal:
    id: int
    negated: bool


@dataclass
class Clause:
    literals: list[Literal]


class Cnf(Problem):
    variables: set[int]
    clauses: list[Clause]

    def __init__(self, variables: set[int], clauses: list[Clause]):
        self.variables = variables
        self.clauses = clauses

    def __repr__(self) -> str:
        return f"Cnf(variables={self.variables}, clauses={self.clauses})"

    def cnf(self) -> "Cnf":
        """
        Return self, as this object is already a CNF formula.

        This implementation satisfies the Problem interface for raw CNF inputs,
        allowing the solver to handle them transparently without casting.
        """
        return self

    @classmethod
    def parse(cls, text: str):
        num_variables = -1
        num_clauses = -1
        variables = set()
        clauses = []
        for i in cnf_lines(text):
            if type(i) is tuple:
                if num_variables != -1 or num_clauses != -1:
                    raise ValueError("cnf: problem line appears multiple times")
                num_variables = i[0]
                num_clauses = i[1]
            else:
                if num_variables == -1 or num_clauses == -1:
                    raise ValueError("cnf: clause appeared before problem line")
                for lit in i.literals:
                    variables.add(lit.id)
                clauses.append(i)
        assert len(variables) == num_variables
        assert len(clauses) == num_clauses
        return cls(variables, clauses)


def cnf_lines(text: str):
    for line in text.splitlines():
        line = line.strip()
        match line.split():
            case ["c", *_args]:
                # ignore comments
                continue
            case [("%" | "0" | "")] | []:
                continue
            case ["p", "cnf", num_vars, num_clauses]:
                num_vars = int(num_vars)
                num_clauses = int(num_clauses)
                yield (num_vars, num_clauses)
            case clause:
                assert clause[-1] == "0"
                literals = []
                for literal in clause[:-1]:
                    literal = int(literal)
                    assert literal != 0
                    literals.append(Literal(abs(literal), literal < 0))
                yield Clause(literals)
