from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ...enum import RpFitResult
from ...type import LastFitData


@dataclass(frozen=True)
class RpGridResult:
    optimal_fits: tuple[LastFitData, ...]
    fit_result: RpFitResult
    rp_diff: npt.NDArray[np.float64]
    solve_time: float

    @property
    def fit(self) -> LastFitData:
        return self.optimal_fits[0]

    @property
    def optimal_fit_count(self) -> int:
        return len(self.optimal_fits)

    @property
    def has_multiple_solutions(self) -> bool:
        return self.optimal_fit_count > 1

    @property
    def loss(self) -> float:
        return float(np.mean(self.rp_diff ** 2))

    @property
    def is_exact(self) -> bool:
        return not np.count_nonzero(self.rp_diff)
