from types import SimpleNamespace
from typing import Generator

import numpy as np
from numpy import typing as npt

from ..fit import get_rate_combo_fit_result
from ..typedef import OptimizerFitContext, RateComboFitResult
from ...const import TICK_INTERVAL
from ...type import LastFitData


def generate_adjacent_points(base: LastFitData, /, gap: float) -> Generator[LastFitData, None, None]:
    for point in [
        LastFitData(ing=base.ing, skl=base.skl + gap),
        LastFitData(ing=base.ing + gap, skl=base.skl),
        LastFitData(ing=base.ing, skl=base.skl - gap),
        LastFitData(ing=base.ing - gap, skl=base.skl),
    ]:
        if point.is_in_boundary():
            yield point


def traverse_last_fit_dfs(
    center: LastFitData,
    context: OptimizerFitContext,
    reference_rp: npt.NDArray[np.float64],
    computed: SimpleNamespace,
) -> RateComboFitResult:
    visited = set()

    current_fit: RateComboFitResult = get_rate_combo_fit_result(
        context,
        center,
        reference_rp,
        computed,
        initiator="Solve",
        print_non_regular_result_only=True,
    )
    prev_fit: RateComboFitResult | None = None

    while True:
        if current_fit.fit in visited:
            # Already visited, so it's going backward, break the loop
            break

        visited.add(current_fit.fit)

        if current_fit.loss == 0 or (prev_fit is not None and current_fit.is_other_preferred(prev_fit)):
            # Early-terminate if the current MSE becomes 0 (perfect fit)
            # or > current MSE > previous MSE (gets worse)
            break

        best_adjacent: RateComboFitResult | None = None

        for point in generate_adjacent_points(current_fit.fit, gap=TICK_INTERVAL):
            current_adjacent: RateComboFitResult = get_rate_combo_fit_result(
                context,
                point,
                reference_rp,
                computed,
                initiator="Solve",
                print_non_regular_result_only=True,
            )

            if best_adjacent is None or best_adjacent.is_other_preferred(current_adjacent):
                best_adjacent = current_adjacent

        # If no better adjacent point found, stop (reached optimum)
        if best_adjacent is None or best_adjacent.is_other_preferred(current_fit):
            break

        prev_fit = current_fit
        current_fit = best_adjacent

    return current_fit
