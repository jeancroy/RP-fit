from types import SimpleNamespace
from typing import Callable, Hashable, Literal

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from .const import MAX_POSSIBLE_FITS
from .traverse import traverse_last_fit
from ..calc import compute_rp
from ..enum import RpFitResult
from ..type import LastFitData
from ..utils import remove_nan


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


def get_rate_combo_fit_result(
    pokemon_name: Hashable,
    idx: int | None,
    centroid: LastFitData,
    reference_rp: npt.NDArray[np.float64],
    x0,
    unpack_info,
    data: DataFrame,
    computed: SimpleNamespace,
    /,
    initiator: Literal["Solve", "Validate"],
    print_func: Callable[[str], None]
) -> tuple[LastFitData, RpFitResult]:
    rp_diff = reference_rp - compute_rp(x0, data, computed, unpack_info, fit=centroid)
    # Clean as in NaNs removed
    # NaN can be caused by various reasons, including:
    # - New Pokémon max level released with outdated ingredient growth data
    rp_diff_clean = remove_nan(rp_diff)

    current_fit_result = get_rp_fit_result(rp_diff_clean, lax=True)

    if not current_fit_result.is_possible_fit:
        if idx is not None and idx % 1000 == 0:
            print_func(
                f"{initiator} - Finding rate combo of {pokemon_name:<25} - "
                f"{idx} / {MAX_POSSIBLE_FITS} ({idx / MAX_POSSIBLE_FITS:.2%})"
            )

        return centroid, RpFitResult.FAILED

    if current_fit_result == RpFitResult.SUBOPTIMAL:
        # Check the surrounding of the suboptimal result to see if there is a perfect result
        for surrounding in traverse_last_fit(centroid, max_radius=3):
            surrounding_fit_result = get_rp_fit_result(
                remove_nan(reference_rp - compute_rp(x0, data, computed, unpack_info, fit=surrounding)),
                lax=False
            )
            if surrounding_fit_result != RpFitResult.PERFECT:
                continue

            current_fit_result = surrounding_fit_result
            centroid = surrounding
            break

    # Ensure that there are no multiple perfect results
    if current_fit_result == RpFitResult.PERFECT:
        perfect_fits = [centroid]
        # Check the surrounding of the perfect result to make sure every other fit is not perfect
        for surrounding in traverse_last_fit(centroid, max_radius=2, skip_center=True):
            surrounding_fit_result = get_rp_fit_result(
                remove_nan(reference_rp - compute_rp(x0, data, computed, unpack_info, fit=surrounding)),
                lax=False
            )
            if surrounding_fit_result != RpFitResult.PERFECT:
                continue

            perfect_fits.append(surrounding)

        if len(perfect_fits) > 1:
            print_func(
                f"{initiator} - {pokemon_name:<25} has multiple ({len(perfect_fits)}) perfect fits: "
                f"{" / ".join(f"[Ing {fit.ing:>6.2%} / Skl {fit.skl:>6.2%}]" for fit in perfect_fits)}"
            )
            current_fit_result = RpFitResult.SUBOPTIMAL

    if np.isnan(rp_diff).any():
        print_func(f"{initiator} - WARNING - RP diff of {pokemon_name} has NaN")

    print_func(
        f"{initiator} - [{current_fit_result.name}] RP fit of {pokemon_name:<25} found at: "
        f"Ing {centroid.ing:>6.2%} / Skl {centroid.skl:>6.2%}"
    )
    if current_fit_result == RpFitResult.SUBOPTIMAL:
        print_func(
            f"{" " * (len(initiator) + 3)}RP diff: {rp_diff_clean[rp_diff_clean != 0]} "
            f"({(rp_diff_clean != 0).sum()} / {rp_diff_clean.size} - {pokemon_name})"
        )

    return centroid, current_fit_result
