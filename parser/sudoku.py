"""
Parser for sudoku files.

Each puzzle must be in the following format. It may be of dimensions nxn, where
n is any natural number. The parser will determine n automatically.

                        2 5 . | . . 3 | . . .
                        . . . | . . . | 2 7 .
                        8 7 . | . . 6 | 4 . .
                        ------+-------+------
                        . 2 . | . . 8 | 1 9 3
                        . 1 5 | . 4 . | 8 . .
                        . . . | 1 . . | . . 4
                        ------+-------+------
                        . . . | 7 3 4 | . . .
                        . . . | 6 . . | . . 9
                        . 6 4 | . . 9 | . 5 8

Here, a '.' represents an empty square, and the rest of the squares contain
numbers placed as part of the puzzle. The program will produce a similar grid
with all the squares containing numbers once it solves it.

Also supports converting the Sudoku board into CNF for SAT solvers.
"""

import math
from typing import Dict, List, Optional, Set, Tuple

from .problem import Problem
from .cnf import Cnf, Clause, Literal


class Cell:
    """
    Represents a single cell in a Sudoku grid.

    Attributes:
        x: The row coordinate of the cell (0-indexed).
        y: The column coordinate of the cell (0-indexed).
        number: The solved number in this cell, or -1 if empty.
        choices: The number of possible values remaining for this cell.
        constraints: A bitset where each bit represents whether a number is impossible.
    """

    def __init__(self, x: int, y: int, size: int) -> None:
        """
        Initialize a new cell at the given coordinates.

        Args:
            x: Row position in the grid.
            y: Column position in the grid.
            size: Dimension of the sudoku grid (e.g., 9 for a 9x9 puzzle).
        """
        # store the cell's position in the grid
        self.x, self.y = x, y

        # the number currently placed in this cell, -1 means empty
        self.number = -1

        # initially all numbers are possible, so we have 'size' choices
        self.choices = size

        # bitset where bit i is 1 if number i is impossible, starts with all zeros
        self.constraints = 0

    def __repr__(self) -> str:
        """
        Return a string representation of the cell for debugging.

        Returns:
            A string showing the cell's coordinates and current number.
        """
        return f"Cell(x={self.x}, y={self.y}, number={self.number})"


class Sudoku(Problem):
    """
    Represents a Sudoku puzzle with its grid and operations.

    Attributes:
        size: The dimension of the grid (e.g., 9 for standard Sudoku).
        block: The dimension of each block (e.g., 3 for a 9x9 Sudoku).
        board: A 2D list representing the puzzle, with -1 for empty cells.
    """

    def __init__(self, board: Optional[List[List[int]]] = None) -> None:
        """
        Initialize a Sudoku puzzle from a board.

        Args:
            board: A 2D list of integers representing the puzzle state.
        """
        # we can initialize from a pre-built board or default to empty 9x9
        if board:
            # if we have a board, infer the size from its dimensions and note
            # that the block size is the square root of the size.
            self.size = len(board)
            self.block = int(math.isqrt(self.size))
            self.board = board
        else:
            # default to standard 9x9 sudoku if nothing is provided
            self.size, self.block = 9, 3
            # create an empty 9x9 grid filled with -1 values
            self.board = [[-1 for _ in range(9)] for _ in range(9)]

    def __repr__(self) -> str:
        """
        Return a string representation of the Sudoku object.

        Returns:
            A string showing the size and board state.
        """
        return f"Sudoku(size={self.size}, board={self.board})"

    @classmethod
    def parse(cls, content: str) -> "Sudoku":
        """
        Parse a text representation of a Sudoku puzzle.

        This method extracts alphanumeric characters and dots from the input,
        determines the grid size automatically, and populates the board.

        Args:
            content: String containing the puzzle, where '.' represents empty cells.

        Returns:
            A new Sudoku instance with the parsed board.

        Raises:
            ValueError: If the number of cells doesn't form a perfect square grid.
        """
        # extract only the meaningful characters: digits, letters, and dots
        # this ignores formatting characters like pipes, dashes, and spaces
        tokens = [c for c in content if c.isalnum() or c == "."]

        # count how many cells we have and compute the grid dimension
        cells = len(tokens)
        # for a valid sudoku, the total number of cells must be a perfect square
        size = int(math.isqrt(cells))

        # verify that we have exactly size * size cells, not more or less
        if size * size != cells:
            raise ValueError(f"sudoku: {cells} cells do not form a square grid.")

        # initialize the board with empty cells (-1)
        board = [[-1 for _ in range(size)] for _ in range(size)]

        # iterate through each token and place it in the appropriate position
        for idx, token in enumerate(tokens):
            # convert the linear index to 2d coordinates
            row = idx // size
            col = idx % size

            if token == ".":
                # dot means empty, which is -1 in our representation
                board[row][col] = -1
            else:
                # convert the character to an internal number (0-indexed)
                if token.isdigit():
                    # digits 1-9 map to internal values 0-8
                    board[row][col] = int(token) - 1
                else:
                    # for puzzles larger than 9x9, we use letters
                    # 'A'/'a' maps to 9, 'B'/'b' to 10, etc.
                    board[row][col] = ord(token.upper()) - 55

        # create and return a new Sudoku instance
        return cls(board=board)

    def display(self) -> None:
        """
        Print the Sudoku grid to the console with formatted blocks.

        This method displays the puzzle with visual separators between blocks,
        using digits 1-9 and letters A-Z for larger puzzles.
        """
        # iterate through each row of the grid
        for i in range(self.size):
            # iterate through each column in the current row
            for j in range(self.size):
                val = self.board[i][j]

                # determine what character to display for this cell
                char = "."
                if val > -1:
                    # convert internal representation back to display character
                    if val < 9:
                        # values 0-8 display as 1-9
                        char = str(val + 1)
                    else:
                        # values 9+ display as A, B, C, etc.
                        char = chr(val + 55)

                # print the character followed by a space
                print(f"{char}", end=" ")

                # add vertical separator between blocks (but not at the end)
                if (j + 1) % self.block == 0 and j < self.size - 1:
                    print("|", end=" ")

            # move to the next line after finishing a row
            print()

            # add horizontal separator between blocks (but not at the end)
            if (i + 1) % self.block == 0 and i < self.size - 1:
                # calculate how many dashes we need based on grid size
                # each cell takes 2 chars (digit + space), plus separators
                dash_count = (self.size * 2) + (self.block - 1) * 2
                print("-" * dash_count)

    @property
    def assignment(self) -> Dict[int, bool]:
        """
        Convert the Sudoku solution to a dictionary of variable assignments.

        This maps each (row, col, value) tuple to a boolean indicating whether
        that value is placed in that cell. Uses the same variable encoding as
        the cnf() method to ensure compatibility with SAT solver verification.

        Returns:
            A dictionary mapping variable IDs to boolean values.

        Raises:
            ValueError: If the board contains any empty cells (value -1).
        """
        n = self.size
        assignment: Dict[int, bool] = {}

        # helper to map (r, c, v) to dimacs variable (1-indexed), matching the
        # encoding used in cnf() method so verification works correctly
        def var(r: int, c: int, v: int) -> int:
            return (r * n * n) + (c * n) + (v + 1)

        # iterate through each cell in the board
        for r in range(n):
            for c in range(n):
                val = self.board[r][c]

                # check if this cell is empty, which means the puzzle isn't solved
                if val == -1:
                    raise ValueError(
                        f"sudoku: cannot create assignment from incomplete board "
                        f"(empty cell at row {r}, col {c})"
                    )

                # for this cell, set the correct value to true and all others to false
                for v in range(n):
                    var_id = var(r, c, v)
                    # the variable is true only if this is the value in the cell
                    assignment[var_id] = v == val

        return assignment

    def cnf(self) -> Cnf:
        """
        Convert the Sudoku puzzle to CNF using optimized extended encoding.

        This implements the optimization from the paper:
        "Optimized CNF Encoding for Sudoku Puzzles" by Kwon & Jain
        https://www.cs.cmu.edu/~hjain/papers/sudoku-as-SAT.pdf

        The key insight is to use the fixed cells (givens) to eliminate
        clauses and literals before building the full CNF formula. This
        significantly reduces the size of the resulting SAT problem.

        Returns:
            A Cnf object representing the puzzle as a SAT formula.
        """
        n = self.size
        b = self.block

        # build sets of variables we know are true or false from the givens,
        # where v_plus contains (row, col, val) tuples that must be true, while
        # v_minus contains all the tuples that must be false
        v_plus: Set[Tuple[int, int, int]] = set()
        v_minus: Set[Tuple[int, int, int]] = set()

        # helper to calculate the starting coordinate of a block given a position
        def get_block_start(pos: int) -> int:
            return (pos // b) * b

        # helper to map (r, c, v) a to dimacs variable (1-indexed) using a
        # standard encoding: flatten the 3d space into a 1d variable number
        # formula ensures each (r, c, v) maps to a unique positive integer
        def var(r: int, c: int, v: int) -> int:
            return (r * n * n) + (c * n) + (v + 1)

        # analyze all fixed cells to determine which variables must be true/false
        for r in range(n):
            for c in range(n):
                val = self.board[r][c]
                if val != -1:
                    # this exact variable must be true
                    v_plus.add((r, c, val))

                    # by sudoku rules, several other variables must be false:
                    # this cell cannot have any other value (same cell constraint)
                    for v in range(n):
                        if v != val:
                            v_minus.add((r, c, v))
                    # no other cell in this row can have this value (row constraint)
                    for other_c in range(n):
                        if other_c != c:
                            v_minus.add((r, other_c, val))
                    # no other cell in this column can have this value (column constraint)
                    for other_r in range(n):
                        if other_r != r:
                            v_minus.add((other_r, c, val))
                    # no other cell in this block can have this value (block constraint)
                    block_r_start = get_block_start(r)
                    block_c_start = get_block_start(c)
                    for br in range(block_r_start, block_r_start + b):
                        for bc in range(block_c_start, block_c_start + b):
                            # skip the cell itself
                            if br != r or bc != c:
                                v_minus.add((br, bc, val))

        # we'll collect all clauses that survive the optimization
        clauses: List[Clause] = []

        def tautology(literals: List[Tuple[int, int, int, bool]]) -> bool:
            """
            Check if a clause is trivially satisfied by the givens.

            This implements the logic from the paper, which is if any literal
            in the clause is known to be true, the entire clause is satisfied.

            Args:
                literals: List of (row, col, val, negated) tuples.

            Returns:
                True if the clause is satisfied by V+ or V-, False otherwise.
            """
            for r, c, v, negated in literals:
                # a positive literal in v_plus makes the clause true
                if (r, c, v) in v_plus and not negated:
                    return True
                # a negative literal in v_minus makes the clause true
                if (r, c, v) in v_minus and negated:
                    return True
            return False

        def reduce_clause(literals: List[Tuple[int, int, int, bool]]) -> List[Literal]:
            """
            Remove literals that are known to be false from a clause.

            This implements the logic from the paper, which eliminates
            literals that cannot possibly be true based on the givens.

            Args:
                literals: List of (row, col, val, negated) tuples.

            Returns:
                A list of Literal objects with false literals removed.
            """
            reduced: List[Literal] = []
            for r, c, v, negated in literals:
                # a positive literal in v_minus is definitely false
                if (r, c, v) in v_minus and not negated:
                    continue
                # a negative literal in v_plus is definitely false
                if (r, c, v) in v_plus and negated:
                    continue
                # this literal might be true, so keep it
                reduced.append(Literal(var(r, c, v), negated))
            return reduced

        def add_clause(literals: List[Tuple[int, int, int, bool]]) -> None:
            """
            Add a clause to the CNF after applying optimizations.

            This combines both reduction operators: first check if the clause
            is trivially true (can be deleted), then remove false literals.

            Args:
                literals: List of (row, col, val, negated) tuples representing the clause.
            """
            # if the clause is already satisfied, don't add it at all
            if tautology(literals):
                return

            # remove literals that are known to be false
            reduced = reduce_clause(literals)

            # only add the clause if it still has literals after reduction
            if reduced:
                clauses.append(Clause(reduced))

        # each cell must have at least one value
        for r in range(n):
            for c in range(n):
                # the clause is a disjunction of all possible values for this cell
                lits = [(r, c, v, False) for v in range(n)]
                add_clause(lits)

        # each cell can have at most one value
        for r in range(n):
            for c in range(n):
                # for every pair of values, they can't both be true
                for v1 in range(n):
                    for v2 in range(v1 + 1, n):
                        add_clause([(r, c, v1, True), (r, c, v2, True)])

        # each row must contain at least one of each number
        for r in range(n):
            for v in range(n):
                # value v must appear somewhere in row r
                lits = [(r, c, v, False) for c in range(n)]
                add_clause(lits)

        # each row can contain at most one of each number
        for r in range(n):
            for v in range(n):
                # value v can't appear in two different columns of the same row
                for c1 in range(n):
                    for c2 in range(c1 + 1, n):
                        add_clause([(r, c1, v, True), (r, c2, v, True)])

        # each column must contain at least one of each number
        for c in range(n):
            for v in range(n):
                # value v must appear somewhere in column c
                lits = [(r, c, v, False) for r in range(n)]
                add_clause(lits)

        # each column can contain at most one of each number
        for c in range(n):
            for v in range(n):
                # value v can't appear in two different rows of the same column
                for r1 in range(n):
                    for r2 in range(r1 + 1, n):
                        add_clause([(r1, c, v, True), (r2, c, v, True)])

        # each block must contain at least one of each number
        for block_idx in range(n):
            # calculate the top-left corner of this block
            block_r_start = (block_idx // b) * b
            block_c_start = (block_idx % b) * b
            for v in range(n):
                # value v must appear somewhere in this block
                lits = []
                for br in range(b):
                    for bc in range(b):
                        lits.append((block_r_start + br, block_c_start + bc, v, False))
                add_clause(lits)

        # each block can contain at most one of each number
        for block_idx in range(n):
            # calculate the top-left corner of this block
            block_r_start = (block_idx // b) * b
            block_c_start = (block_idx % b) * b
            # collect all cell coordinates in this block
            cells: List[Tuple[int, int]] = []
            for br in range(b):
                for bc in range(b):
                    cells.append((block_r_start + br, block_c_start + bc))

            for v in range(n):
                # value v can't appear in two different cells of the same block
                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        add_clause(
                            [
                                (cells[i][0], cells[i][1], v, True),
                                (cells[j][0], cells[j][1], v, True),
                            ]
                        )

        # return the complete formula with all variables and clauses
        return Cnf(set(range(n**3)), clauses)
