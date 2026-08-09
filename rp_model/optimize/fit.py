from types import SimpleNamespace

import numpy as np
import numpy.typing as npt

from .typedef import OptimizerFitContext
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

    return RpFitResult.FAILED


def get_rp_diff(
    context: OptimizerFitContext,
    reference_rp: npt.NDArray[np.float64],
    computed: SimpleNamespace,
    fit_data: LastFitData,
) -> npt.NDArray[np.float64]:
    return remove_nan(reference_rp - compute_rp(
        context.x0,
        context.pokemon_data_of_group,
        computed,
        context.unpack_info,
        fit=fit_data,
    ))
