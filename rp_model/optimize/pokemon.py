from typing import Callable, Hashable

from numpy import float64

from .fit import get_rate_combo_fit_result
from .traverse.bfs import traverse_last_fit_bfs
from .typedef import OptimizerSingleFitResult, OptimizerSolvedDataEntry
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
) -> OptimizerSolvedDataEntry:
    last_fit_of_pokemon = last_fit_dict.get(pokemon_name, LastFitData.default())

    computed = make_precomputed_columns(pokemon_data_of_group)
    reference_rp = pokemon_data_of_group["RP"].astype(float64).to_numpy()

    single_mon_fit_results: set[OptimizerSingleFitResult] = set()
    # Expand 2 layers with each having gaps of 0.03 (3%) for starting from different centroids
    # to make sure that there isn't a duplicated perfect fit
    for centroid in traverse_last_fit_bfs(last_fit_of_pokemon, max_radius=3, point_gap=0.03):
        for idx, fit_data in enumerate(traverse_last_fit_bfs(centroid)):
            actual_rate_combo, fit_result = get_rate_combo_fit_result(
                pokemon_name,
                idx,
                fit_data,
                reference_rp,
                x0,
                unpack_info,
                pokemon_data_of_group,
                computed,
                initiator="Solve",
                print_func=print_func
            )

            single_fit_result = OptimizerSingleFitResult(
                ing=actual_rate_combo.ing,
                skl=actual_rate_combo.skl,
                result=fit_result,
            )

            single_mon_fit_results.add(single_fit_result)
            if RP_MODEL_IS_GLOBAL_CHECK or fit_result != RpFitResult.PERFECT:
                # Keep recording fits if the result is not failed
                continue

            single_mon_fit_results = {single_fit_result}
            break

    solutions_found: list[OptimizerSingleFitResult] = list(filter(
        lambda x: x.result.is_possible_fit, single_mon_fit_results
    ))
    solutions_formatted = [
        f"[Ing] {solution.ing:.2%} [Skl] {solution.skl:.2%} ({solution.result.name[:1]})"
        for solution in solutions_found
    ]
    print_func(f"{pokemon_name:>25} - {len(solutions_found)} solutions found - {" / ".join(solutions_formatted)}")
    try:
        fit_result_to_use_for_mon = sorted(single_mon_fit_results, key=lambda x: x.result.value, reverse=True)[0]

        if fit_result_to_use_for_mon.result != RpFitResult.PERFECT:
            imperfect = fit_result_to_use_for_mon

            print_func(
                f"WARNING - Imperfect solution used for [{pokemon_name}] - "
                f"Ingredient: {imperfect.ing:>6.2%} / Skill: {imperfect.skl:>6.2%}"
            )

        return OptimizerSolvedDataEntry(
            ing=fit_result_to_use_for_mon.ing,
            skl=fit_result_to_use_for_mon.skl,
            result=fit_result_to_use_for_mon.result,
            pokemon=pokemon_name,
        )
    except IndexError:
        print_func(f"WARNING - No solution found for [{pokemon_name}], default is used")
        default = LastFitData.default()

        return OptimizerSolvedDataEntry(
            ing=default.ing,
            skl=default.skl,
            result=RpFitResult.FAILED,
            pokemon=pokemon_name,
        )
