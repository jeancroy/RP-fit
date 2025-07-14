import math
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

import numpy as np
import numpy.typing as npt
from numpy import float64
from pandas import DataFrame

from ..enum import RpFitResult
from ..type import LastFitData

data_rate_comparison_tolerance = 1e-6  # Adjust this value as needed


@dataclass
class OptimizerSingleFitResult:
    fit: LastFitData
    result: RpFitResult

    def __eq__(self, other) -> bool:
        if not isinstance(other, OptimizerSingleFitResult):
            return False

        return (
                self.result == other.result and
                math.isclose(self.fit.ing, other.fit.ing, abs_tol=data_rate_comparison_tolerance) and
                math.isclose(self.fit.skl, other.fit.skl, abs_tol=data_rate_comparison_tolerance)
        )

    def __hash__(self) -> int:
        # Round the float values to ensure consistent hashing for values within tolerance
        precision = int(-math.log10(data_rate_comparison_tolerance))
        rounded_ing = round(self.fit.ing, precision)
        rounded_skl = round(self.fit.skl, precision)

        return hash((rounded_ing, rounded_skl, self.result))


@dataclass
class RateComboFitResult(OptimizerSingleFitResult):
    rp_diff: npt.NDArray[float64]

    # >= 0, the less, the better with 0 meaning perfect match
    loss: np.floating = field(init=False)

    def __post_init__(self):
        self.loss = np.mean(self.rp_diff ** 2) # MSE

    def is_other_preferred(self, other: "RateComboFitResult") -> bool:
        if self.loss == 0:
            return False

        if other.loss == 0:
            return True

        return self.loss > other.loss

    def __eq__(self, other) -> bool:
        if not isinstance(other, RateComboFitResult):
            return False

        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class OptimizerSolvedDataEntry(OptimizerSingleFitResult):
    pokemon: Hashable

    def __eq__(self, other) -> bool:
        if not isinstance(other, OptimizerSolvedDataEntry):
            return False

        return self.pokemon == other.pokemon and self.result == other.result

    def __hash__(self) -> int:
        return hash((self.pokemon, self.result))


@dataclass
class OptimizerFitContext:
    x0: Any
    unpack_info: Any
    pokemon_name: Hashable
    pokemon_data_of_group: DataFrame
    print_func: Callable[[str], None]
