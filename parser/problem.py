from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cnf import Cnf


class Problem(ABC):
    """
    Abstract base class for all problem types.

    This interface ensures that any problem domain (like Sudoku) can provide
    a standard CNF representation, allowing general SAT solvers to process it.
    """

    @abstractmethod
    def cnf(self) -> "Cnf":
        """
        Convert the problem instance into a CNF formula.

        Returns:
            A Cnf object representing the boolean constraints of the problem.
        """
        pass
