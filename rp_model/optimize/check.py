from types import SimpleNamespace
from typing import Literal

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from .traverse import MAX_POSSIBLE_FITS, traverse_last_fit
from ..calc import compute_rp
from ..enum import RpFitResult
from ..type import LastFitData
from ..utils import remove_nan


def get_rp_fit_result(rp_diff_clean: npt.NDArray[np.float64]) -> RpFitResult:
    if not rp_diff_clean.any():
        return RpFitResult.PERFECT

    # Allow small rounding error (1 per 10 data) due to whatever reason
    if rp_diff_clean.min() >= -1 and rp_diff_clean.max() <= 1 and abs(rp_diff_clean.mean()) < 0.1:
        return RpFitResult.SUBOPTIMAL

    return RpFitResult.FAILED


def is_rate_combo_fit(
    pokemon_name: str,
    idx: int | None,
    rate_combo: LastFitData,
    reference_rp: npt.NDArray[np.float64],
    x0,
    unpack_info,
    data: DataFrame,
    computed: SimpleNamespace,
    /,
    initiator: Literal["Solve", "Validate"]
) -> bool:
    rp_diff = reference_rp - compute_rp(x0, data, computed, unpack_info, fit=rate_combo)
    # Clean as in NaNs removed
    # NaN can be caused by various reasons, including:
    # - New Pokémon max level released with outdated ingredient growth data
    rp_diff_clean = remove_nan(rp_diff)

    current_fit_result = get_rp_fit_result(rp_diff_clean)

    if not current_fit_result.is_possible_fit:
        if idx is not None and idx % 1000 == 0:
            print(
                f"{initiator} - Finding rate combo of {pokemon_name:<15} - "
                f"{idx} / {MAX_POSSIBLE_FITS} ({idx / MAX_POSSIBLE_FITS:.2%})"
            )

        return False

    if current_fit_result == RpFitResult.SUBOPTIMAL:
        # Check the surrounding of the suboptimal result to see if there is a perfect result
        for surrounding in traverse_last_fit(rate_combo, max_radius=3):
            surrounding_fit_result = get_rp_fit_result(
                remove_nan(reference_rp - compute_rp(x0, data, computed, unpack_info, fit=surrounding))
            )
            if surrounding_fit_result != RpFitResult.PERFECT:
                continue

            current_fit_result = surrounding_fit_result
            rate_combo = surrounding
            break

    # After suboptimal result check, report the reason of suboptimal if it doesn't flip to perfect
    if current_fit_result == RpFitResult.SUBOPTIMAL:
        print(
            f"{initiator} - {pokemon_name:<15} has suboptimal fit due to the following RP diff: "
            f"{rp_diff_clean[rp_diff_clean != 0]} ({rp_diff_clean.size})"
        )

    # Ensure that there are no multiple perfect results
    if current_fit_result == RpFitResult.PERFECT:
        perfect_fits = [rate_combo]
        # Check the surrounding of the perfect result to make sure every other fit is not perfect
        for surrounding in traverse_last_fit(rate_combo, max_radius=2, skip_center=True):
            surrounding_fit_result = get_rp_fit_result(
                remove_nan(reference_rp - compute_rp(x0, data, computed, unpack_info, fit=surrounding))
            )
            if surrounding_fit_result != RpFitResult.PERFECT:
                continue

            perfect_fits.append(surrounding)

        if len(perfect_fits) > 1:
            print(
                f"{initiator} - {pokemon_name:<15} has multiple ({len(perfect_fits)}) perfect fits: "
                f"{" / ".join(f"[Ing {fit.ing:>6.2%} / Skl {fit.skl:>6.2%}]" for fit in perfect_fits)}"
            )
            current_fit_result = RpFitResult.SUBOPTIMAL

    if np.isnan(rp_diff).any():
        print(f"{initiator} - WARNING - RP diff of {pokemon_name} has NaN")

    print(
        f"{initiator} - [{current_fit_result.name}] RP fit of {pokemon_name:<15} found at: "
        f"Ing {rate_combo.ing:>6.2%} / Skl {rate_combo.skl:>6.2%}"
    )
    return True
