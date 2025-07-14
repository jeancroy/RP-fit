from types import SimpleNamespace
from typing import Hashable, Literal

import numpy as np
import numpy.typing as npt

from .traverse.bfs import traverse_last_fit_bfs
from .typedef import OptimizerFitContext, RateComboFitResult
from ..calc import compute_rp
from ..const import MAX_POSSIBLE_FITS
from ..enum import RpFitResult
from ..type import LastFitData
from ..utils import remove_nan

_cache: dict[tuple[Hashable, LastFitData], RateComboFitResult] = {}


def get_rp_fit_result(rp_diff_clean: npt.NDArray[np.float64], /, lax: bool) -> RpFitResult:
    if not rp_diff_clean.any():
        return RpFitResult.PERFECT

    if rp_diff_clean.min() >= -1 and rp_diff_clean.max() <= 1:
        avg = abs(rp_diff_clean).mean()

        # Allow small rounding error (1 per 50 data) likely due to rounding reason
        if lax and avg < 1 / 50:
            return RpFitResult.PERFECT

        # If not lax, and the error rate is < 1 per 30 data, count it as suboptimal
        if avg < 1 / 30:
            return RpFitResult.SUBOPTIMAL

        # Otherwise, count it as failed
        return RpFitResult.FAILED

    return RpFitResult.FAILED


def get_rp_diff(
    context: OptimizerFitContext,
    reference_rp: npt.NDArray[np.float64],
    computed: SimpleNamespace,
    fit_data: LastFitData,
):
    # Clean as in NaNs removed
    # NaN can be caused by various reasons, including:
    # - New Pokémon max level released with outdated ingredient growth data
    return remove_nan(reference_rp - compute_rp(
        context.x0,
        context.pokemon_data_of_group,
        computed,
        context.unpack_info,
        fit=fit_data
    ))


def get_rate_combo_fit_result(
    context: OptimizerFitContext,
    fit_data: LastFitData,
    reference_rp: npt.NDArray[np.float64],
    computed: SimpleNamespace,
    /,
    initiator: Literal["Solve", "Validate"],
    idx: int | None = None,
    print_non_regular_result_only: bool = False,
) -> RateComboFitResult:
    if result := _cache.get((context.pokemon_name, fit_data)):
        return result

    rp_diff = get_rp_diff(context, reference_rp, computed, fit_data)

    current_fit_result = get_rp_fit_result(rp_diff, lax=True)

    if not current_fit_result.is_possible_fit:
        if idx is not None and idx % 3000 == 0:
            context.print_func(
                f"{initiator} - Finding rate combo of {context.pokemon_name:<25} - "
                f"{idx} / {MAX_POSSIBLE_FITS} ({idx / MAX_POSSIBLE_FITS:.2%})"
            )

        result = RateComboFitResult(fit=fit_data, result=RpFitResult.FAILED, rp_diff=rp_diff)

        _cache[(context.pokemon_name, fit_data)] = result
        return result

    if current_fit_result == RpFitResult.SUBOPTIMAL:
        # Check the surrounding of the suboptimal result to see if there is a perfect result
        for surrounding in traverse_last_fit_bfs(fit_data, max_radius=3):
            surrounding_fit_result = get_rp_fit_result(
                get_rp_diff(context, reference_rp, computed, surrounding),
                lax=False
            )
            if surrounding_fit_result != RpFitResult.PERFECT:
                continue

            current_fit_result = surrounding_fit_result
            fit_data = surrounding
            break

    # Ensure that there are no multiple perfect results
    if current_fit_result == RpFitResult.PERFECT:
        perfect_fits = [fit_data]
        # Check the surrounding of the perfect result to make sure every other fit is not perfect
        for surrounding in traverse_last_fit_bfs(fit_data, max_radius=2, skip_center=True):
            surrounding_fit_result = get_rp_fit_result(
                get_rp_diff(context, reference_rp, computed, surrounding),
                lax=False
            )
            if surrounding_fit_result != RpFitResult.PERFECT:
                continue

            perfect_fits.append(surrounding)

        if len(perfect_fits) > 1:
            context.print_func(
                f"{initiator} - {context.pokemon_name:<25} has multiple ({len(perfect_fits)}) perfect fits: "
                f"{" / ".join(f"[Ing {fit.ing:>6.2%} / Skl {fit.skl:>6.2%}]" for fit in perfect_fits)}"
            )
            current_fit_result = RpFitResult.SUBOPTIMAL

    if np.isnan(rp_diff).any():
        context.print_func(f"{initiator} - WARNING - RP diff of {context.pokemon_name} has NaN")

    if not print_non_regular_result_only:
        context.print_func(
            f"{initiator} - [{current_fit_result.name}] RP fit of {context.pokemon_name:<25} found at: "
            f"Ing {fit_data.ing:>6.2%} / Skl {fit_data.skl:>6.2%}"
        )
    if current_fit_result == RpFitResult.SUBOPTIMAL:
        context.print_func(
            f"{" " * (len(initiator) + 3)}RP diff: {rp_diff[rp_diff != 0]} "
            f"({(rp_diff != 0).sum()} / {rp_diff.size} - {context.pokemon_name})"
        )

    result = RateComboFitResult(fit=fit_data, result=current_fit_result, rp_diff=rp_diff)

    _cache[(context.pokemon_name, fit_data)] = result
    return result
