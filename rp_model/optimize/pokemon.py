from typing import Callable, Hashable

from numpy import float64

from .fit import get_rate_combo_fit_result
from .traverse import traverse_last_fit
from .typedef import OptimizerSingleFitResult
from ..calc import make_precomputed_columns
from ..enum import RpFitResult
from ..env import RP_MODEL_IS_GLOBAL_CHECK
from ..type import LastFitData


def process_pokemon(
    last_fit_dict,
    x0,
    unpack_info,
    pokemon_name: Hashable,
    pokemon_data_of_group,
    print_func: Callable[[str], None],
):
    last_fit_of_pokemon = last_fit_dict.get(pokemon_name, LastFitData.default())

    computed = make_precomputed_columns(pokemon_data_of_group)
    reference_rp = pokemon_data_of_group["RP"].astype(float64).to_numpy()

    single_mon_fit_results: list[OptimizerSingleFitResult] = []
    for idx, last_fit in enumerate(traverse_last_fit(last_fit_of_pokemon)):
        actual_rate_combo, fit_result = get_rate_combo_fit_result(
            pokemon_name,
            idx,
            last_fit,
            reference_rp,
            x0,
            unpack_info,
            pokemon_data_of_group,
            computed,
            initiator="Solve",
            print_func=print_func
        )

        single_fit_result: OptimizerSingleFitResult = {
            "ing": actual_rate_combo.ing,
            "skl": actual_rate_combo.skl,
            "result": fit_result
        }

        single_mon_fit_results.append(single_fit_result)
        if RP_MODEL_IS_GLOBAL_CHECK or fit_result != RpFitResult.PERFECT:
            # Keep recording fits if the result is not failed
            continue

        single_mon_fit_results = [single_fit_result]
        break

    if RP_MODEL_IS_GLOBAL_CHECK:
        for fit_result in single_mon_fit_results:
            if fit_result["result"] == RpFitResult.FAILED:
                continue

            print_func(f"{pokemon_name}: {fit_result}")

    try:
        fit_result_to_use_for_mon = sorted(
            single_mon_fit_results,
            key=lambda x: x["result"].value, reverse=True
        )[0]

        if fit_result_to_use_for_mon["result"] != RpFitResult.PERFECT:
            print_func(f"WARNING - Imperfect solution used for [{pokemon_name}] - {fit_result_to_use_for_mon}")

        return fit_result_to_use_for_mon | {"pokemon": pokemon_name}
    except IndexError:
        print_func(f"WARNING - No solution found for [{pokemon_name}], default is used")
        return {
            "pokemon": pokemon_name,
            "ing": 0.2,
            "skl": 0.02,
            "result": RpFitResult.FAILED
        }
