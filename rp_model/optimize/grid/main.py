from time import perf_counter

import numpy as np

from .type import RpGridResult
from ..component import get_rp_rate_components
from ..fit import get_rp_fit_result
from ..rate import get_ingredient_rate_ticks, get_skill_rate_ticks
from ..typedef import OptimizerFitContext
from ...calc import compute_rp, make_precomputed_columns
from ...enum import RpFitResult
from ...type import LastFitData
from ...utils import remove_nan, round_precise


def solve_rp_grid(context: OptimizerFitContext) -> RpGridResult:
    started_at = perf_counter()
    data = context.pokemon_data_of_group
    if data.empty:
        raise ValueError(f"RP grid input for {context.pokemon_name} is empty")

    reference_float = data["RP"].astype(np.float64).to_numpy()
    reference_rp = reference_float.astype(np.int64)
    if not np.array_equal(reference_rp, reference_float):
        raise ValueError(f"RP observations for {context.pokemon_name} must be integers")

    computed = make_precomputed_columns(data)
    ingredient_rates = get_ingredient_rate_ticks()
    skill_rates = get_skill_rate_ticks()
    ingredient_values, skill_values, bonus_multipliers = get_rp_rate_components(
        data,
        computed,
        ingredient_rates,
        skill_rates,
    )

    squared_error = np.zeros((ingredient_rates.size, skill_rates.size), dtype=np.int64)
    for observation_idx, observed_rp in enumerate(reference_rp):
        predicted_rp = round_precise(
            bonus_multipliers[observation_idx] * (
                ingredient_values[:, observation_idx, np.newaxis]
                + skill_values[np.newaxis, :, observation_idx]
            )
        ).astype(np.int64)
        difference = observed_rp - predicted_rp
        squared_error += difference * difference

    minimum_squared_error = int(squared_error.min())
    optimal_indices = np.argwhere(squared_error == minimum_squared_error)
    optimal_fits = tuple(
        LastFitData(ing=float(ingredient_rates[ingredient_idx]), skl=float(skill_rates[skill_idx]))
        for ingredient_idx, skill_idx in optimal_indices
    )
    fit = optimal_fits[0]
    rp_diff = np.asarray(remove_nan(reference_float - compute_rp(
        context.x0,
        data,
        computed,
        context.unpack_info,
        fit=fit,
    )), dtype=np.float64)
    replayed_squared_error = int(np.sum(rp_diff ** 2))
    if replayed_squared_error != minimum_squared_error:
        raise ValueError(
            f"RP grid for {context.pokemon_name} disagreed with production replay: "
            f"{minimum_squared_error} != {replayed_squared_error}"
        )

    fit_result = get_rp_fit_result(rp_diff, lax=False)
    if minimum_squared_error == 0 and len(optimal_fits) > 1:
        fit_result = RpFitResult.SUBOPTIMAL
    return RpGridResult(
        optimal_fits=optimal_fits,
        fit_result=fit_result,
        rp_diff=rp_diff,
        solve_time=perf_counter() - started_at,
    )
