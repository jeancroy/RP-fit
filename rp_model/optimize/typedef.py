import math
from dataclasses import dataclass
from typing import Hashable

from ..enum import RpFitResult

data_rate_comparison_tolerance = 1e-6  # Adjust this value as needed


@dataclass
class OptimizerSingleFitResult:
    ing: float
    skl: float
    result: RpFitResult

    def __eq__(self, other) -> bool:
        if not isinstance(other, OptimizerSingleFitResult):
            return False

        return (
                self.result == other.result and
                math.isclose(self.ing, other.ing, abs_tol=data_rate_comparison_tolerance) and
                math.isclose(self.skl, other.skl, abs_tol=data_rate_comparison_tolerance)
        )

    def __hash__(self) -> int:
        # Round the float values to ensure consistent hashing for values within tolerance
        precision = int(-math.log10(data_rate_comparison_tolerance))
        rounded_ing = round(self.ing, precision)
        rounded_skl = round(self.skl, precision)

        return hash((rounded_ing, rounded_skl, self.result))


@dataclass
class OptimizerSolvedDataEntry(OptimizerSingleFitResult):
    pokemon: Hashable

    def __eq__(self, other) -> bool:
        if not isinstance(other, OptimizerSolvedDataEntry):
            return False

        return self.pokemon == other.pokemon and self.result == other.result

    def __hash__(self) -> int:
        return hash((self.pokemon, self.result))
