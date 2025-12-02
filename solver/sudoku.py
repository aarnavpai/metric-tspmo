"""
Solver of Sudoku puzzles.

The program uses constraint propagation combined with recursive backtracking.
Each cell maintains a bitmask of impossible values. Placing a number updates
all cells in the same row, column, and box. If a cell is reduced to a single
possibility, it is filled immediately, triggering propagation of this new
constraint. When propagation stalls, the solver selects the cell with the
fewest remaining choices, guesses a valid number, and attempts to solve the
rest recursively, backtracking if a contradiction arises.
"""

from typing import List, Optional

from parser import Sudoku, Cell


def test_bit(bs: int, idx: int) -> bool:
    """
    Check if a specific bit is set in a bitset.

    Args:
        bs: The bitset to test.
        idx: The bit position to check (0-indexed).

    Returns:
        True if the bit at position idx is 1, False otherwise.
    """
    return (bs & (1 << idx)) != 0


def set_bit(bs: int, idx: int) -> int:
    """
    Set a specific bit to 1 in a bitset.

    Args:
        bs: The bitset to modify.
        idx: The bit position to set (0-indexed).

    Returns:
        A new bitset with the specified bit set to 1.
    """
    return bs | (1 << idx)


def clear_bit(bs: int, idx: int) -> int:
    """
    Clear a specific bit (set it to 0) in a bitset.

    Args:
        bs: The bitset to modify.
        idx: The bit position to clear (0-indexed).

    Returns:
        A new bitset with the specified bit set to 0.
    """
    return bs & ~(1 << idx)


def set_all_bits(n: int) -> int:
    """
    Create a bitset with the first n bits set to 1.

    Args:
        n: The number of bits to set.

    Returns:
        A bitset with bits 0 through n-1 all set to 1.
    """
    return (1 << n) - 1


class SudokuSolver:
    """
    A constraint-propagation based solver for Sudoku puzzles.

    This solver uses a combination of constraint propagation and backtracking
    search. It maintains a bitset of constraints for each cell and propagates
    implications when a cell's value is determined.

    Attributes:
        source: The original Sudoku puzzle to solve.
        size: The dimension of the grid (e.g., 9 for standard Sudoku).
        block: The dimension of each block (e.g., 3 for a 9x9 Sudoku).
    """

    def __init__(self, sudoku: Sudoku) -> None:
        """
        Initialize the solver with a Sudoku puzzle.

        Args:
            sudoku: The Sudoku puzzle to solve.
        """
        self.source = sudoku
        self.size = sudoku.size
        self.block = sudoku.block

    def create_grid(self) -> List[List[Cell]]:
        """
        Allocate a fresh grid of empty Cell objects.

        Returns:
            A 2D list of Cell objects with coordinates set but no values.
        """
        cells: List[List[Cell]] = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append(Cell(i, j, self.size))
            cells.append(row)
        return cells

    def copy_grid(self, source: List[List[Cell]]) -> List[List[Cell]]:
        """
        Create a deep copy of a cell grid.

        This is necessary for backtracking, as we need to restore
        previous states when a guess turns out to be wrong.

        Args:
            source: The grid to copy.

        Returns:
            A new grid with all cell states copied from the source.
        """
        dest = self.create_grid()

        for i in range(self.size):
            for j in range(self.size):
                c = source[i][j]
                d = dest[i][j]

                d.x, d.y = c.x, c.y
                d.number = c.number
                d.choices = c.choices
                d.constraints = c.constraints

        return dest

    def propagate_constraints(
        self, cells: List[List[Cell]], x: int, y: int, hint: int
    ) -> bool:
        """
        Mark a value as impossible for a cell and propagate implications.

        When we determine that a cell cannot have a certain value, we update
        its constraint bitset. If this reduces the cell to only one possible
        value, we immediately fill it and propagate further.

        Args:
            cells: The grid being solved.
            x: Row coordinate of the cell.
            y: Column coordinate of the cell.
            hint: The value (0-indexed) that should be marked impossible.

        Returns:
            True if the constraint is consistent, False if we found a contradiction.
        """
        c = cells[x][y]

        # if this value was already ruled out, we don't need to do anything
        if test_bit(c.constraints, hint):
            return True

        # if the cell is already set to this exact value, we have a contradiction
        # (we're trying to say it can't be what it already is)
        if c.number == hint:
            return False

        # mark this value as impossible by setting its bit in the constraint mask
        # and reduce the count of remaining possibilities
        c.constraints = set_bit(c.constraints, hint)
        c.choices -= 1

        # if there are still multiple possibilities, we're done for now
        if c.choices > 1:
            return True

        # if we've narrowed it down to exactly one choice, find it and fill it
        for val in range(self.size):
            # look for the single value that isn't constrained and process it
            # as a definite hint
            if not test_bit(c.constraints, val):
                return self.process_hint(cells, x, y, val)

        # if we reach here, all values are constrained but nothing is set,
        # which is a contradiction
        return False

    def process_hint(self, cells: List[List[Cell]], x: int, y: int, hint: int) -> bool:
        """
        Set a cell to a specific value and propagate the constraint.

        This is the core of constraint propagation. When we determine a cell's
        value, we must notify all cells in the same row, column, and block that
        they cannot have this value.

        Args:
            cells: The grid being solved.
            x: Row coordinate of the cell.
            y: Column coordinate of the cell.
            hint: The value (0-indexed) to place in the cell.

        Returns:
            True if the hint is consistent with all constraints, False otherwise.
        """
        c: Cell = cells[x][y]

        # if this cell is already set to this value, nothing to do
        if c.number == hint:
            return True

        # if this value was previously ruled out, we have a contradiction
        if test_bit(c.constraints, hint):
            return False

        # lock this cell to the given value by marking all other values impossible
        c.constraints = set_all_bits(self.size)
        c.constraints = clear_bit(c.constraints, hint)
        c.number = hint
        c.choices = 1

        # calculate the top-left corner of the block containing this cell
        br: int = (x // self.block) * self.block
        bc: int = (y // self.block) * self.block

        # propagate this constraint to all peer cells
        # we need to update the same row, same column, and same block
        for i in range(self.size):
            # calculate the coordinates of the i-th cell in this cell's block
            # we iterate through the block in row-major order
            nx: int = br + (i // self.block)
            ny: int = bc + (i % self.block)

            # propagate to the i-th cell in the same row (skip the cell itself)
            if i != x and not self.propagate_constraints(cells, i, y, hint):
                return False

            # propagate to the i-th cell in the same column (skip the cell itself)
            if i != y and not self.propagate_constraints(cells, x, i, hint):
                return False

            # propagate to the i-th cell in the same block (skip the cell itself)
            # we also skip cells we already handled via row/column propagation
            if (nx != x and ny != y) and not self.propagate_constraints(
                cells, nx, ny, hint
            ):
                return False

        # if we made it here, all propagations succeeded
        return True

    def recursively_fill(
        self, cells: List[List[Cell]], empty: List[Cell], k: int
    ) -> bool:
        """
        Attempt to fill remaining cells using backtracking search.

        This function tries all possible values for unsolved cells in order
        of fewest remaining choices (most constrained first). When a guess
        leads to a contradiction, it backtracks and tries the next possibility.

        Args:
            cells: The grid being solved.
            empty: A sorted list of empty cells (by fewest choices first).
            k: The index in the empty list of the next cell to fill.

        Returns:
            True if the puzzle was successfully solved, False if this branch is unsolvable.
        """
        # base case: we've successfully filled all previously-empty cells
        if k >= len(empty):
            return True

        # get the coordinates of the cell we're trying to fill
        x, y = empty[k].x, empty[k].y

        # check if constraint propagation already filled this cell - this can
        # happen when earlier guesses eliminate all but one possibility
        if cells[x][y].number > -1:
            # skip this cell and move to the next one
            return self.recursively_fill(cells, empty, k + 1)

        # snapshot the current state so we can backtrack if needed
        # this is expensive but necessary for correctness
        backtrack = self.copy_grid(cells)

        # try each possible value that hasn't been ruled out
        for val in range(self.size):
            # only try values that are still possible for this cell
            if test_bit(cells[x][y].constraints, val):
                continue

            # attempt to place this value and propagate its constraints
            if self.process_hint(cells, x, y, val):
                # if that succeeded, recursively try to fill the remaining cells
                if self.recursively_fill(cells, empty, k + 1):
                    return True

            # if we get here, this guess led to failure somewhere down the line
            # restore the grid to the state before we made this guess
            for r in range(self.size):
                for c in range(self.size):
                    # overwrite each cell's state with the backup
                    cells[r][c].number = backtrack[r][c].number
                    cells[r][c].choices = backtrack[r][c].choices
                    cells[r][c].constraints = backtrack[r][c].constraints

        # if we tried all values and none worked, this puzzle is unsolvable
        # from this state (backtrack further)
        return False

    def fill_remaining_cells(self, cells: List[List[Cell]]) -> bool:
        """
        Collect all unsolved cells and attempt to fill them.

        This is the entry point for the backtracking search phase. It identifies
        which cells still need values and sorts them by constraint level for
        optimal search performance.

        Args:
            cells: The grid to complete.

        Returns:
            True if the puzzle was solved, False if it's unsolvable.
        """
        # collect all cells that don't have a value yet
        empty: List[Cell] = []
        for i in range(self.size):
            for j in range(self.size):
                if cells[i][j].number == -1:
                    empty.append(cells[i][j])

        # sort by fewest choices first (most constrained variable heuristic)
        # this tends to find contradictions faster and reduces backtracking
        empty.sort(key=lambda c: c.choices)

        # start the recursive backtracking search
        return self.recursively_fill(cells, empty, 0)

    def solve(self) -> Optional[Sudoku]:
        """
        Solve the Sudoku puzzle.

        This is the main entry point for the solver. It creates a cell grid,
        applies the initial constraints from the given cells, and then attempts
        to solve the remaining cells using constraint propagation and backtracking.

        Returns:
            A new Sudoku object with the solution, or None if unsolvable.
        """
        # create a fresh grid with empty cells
        cells = self.create_grid()

        # apply all the initial numbers from the puzzle as constraints
        for i in range(self.size):
            for j in range(self.size):
                val = self.source.board[i][j]
                # skip empty cells
                if val == -1:
                    continue

                # try to apply this given value
                if not self.process_hint(cells, i, j, val):
                    # if a given value creates a contradiction, the puzzle is invalid
                    # print(f"sudoku: found invalid constraint at ({i}, {j}) -> {val + 1}")
                    return None

        # now try to fill in the remaining cells using backtracking
        solved = self.fill_remaining_cells(cells)
        if not solved:
            # if we can't complete it, the puzzle has no solution
            # print("sudoku: unable to fill remaining cells")
            return None

        # extract just the numbers from the solved cell grid
        solution = [
            [cells[i][j].number for j in range(self.size)] for i in range(self.size)
        ]

        # return a new Sudoku object containing the solution
        return Sudoku(board=solution)
