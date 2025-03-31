from typing import TypedDict

from numpy import float64
from pandas import DataFrame

from .check import get_rate_combo_fit_result
from .traverse import traverse_last_fit
from ..calc import make_precomputed_columns
from ..enum import RpFitResult
from ..env import is_pokemon_included_for_rp_model
from ..type import LastFitData


class OptimizerSingleFitResult(TypedDict):
    ing: float
    skl: float
    result: RpFitResult


class OptimizerSolvedDataEntry(TypedDict, OptimizerSingleFitResult):
    pokemon: str


# x0 is a mystery array of numbers that would create different % combinations with `unpack_info`
# x0 + unpack_info = a different combination of rates
def run_optimizer(
    data: DataFrame,
    last_fit_dict: dict[str, LastFitData],
    x0,
    unpack_info
) -> list[OptimizerSolvedDataEntry]:
    solved_data: list[OptimizerSolvedDataEntry] = []

    for pokemon_name, grouped in data.groupby("Pokemon"):
        pokemon_name: str

        if not is_pokemon_included_for_rp_model(pokemon_name):
            continue

        last_fit_of_pokemon = last_fit_dict.get(pokemon_name, LastFitData(ing=0.2, skl=0.02))

        computed = make_precomputed_columns(grouped)
        reference_rp = grouped["RP"].astype(float64).to_numpy()

        single_mon_fit_results: list[OptimizerSingleFitResult] = []
        for idx, last_fit in enumerate(traverse_last_fit(last_fit_of_pokemon)):
            actual_rate_combo, fit_result = get_rate_combo_fit_result(
                pokemon_name,
                idx,
                last_fit,
                reference_rp,
                x0,
                unpack_info,
                grouped,
                computed,
                initiator="Solve"
            )

            single_fit_result: OptimizerSingleFitResult = {
                "ing": actual_rate_combo.ing,
                "skl": actual_rate_combo.skl,
                "result": fit_result
            }

            single_mon_fit_results.append(single_fit_result)
            if fit_result != RpFitResult.PERFECT:
                # Keep recording fits if the result is not failed
                continue

            single_mon_fit_results = [single_fit_result]
            break

        try:
            fit_result_to_use_for_mon = sorted(
                single_mon_fit_results,
                key=lambda x: x["result"].value, reverse=True
            )[0]

            if fit_result_to_use_for_mon["result"] != RpFitResult.PERFECT:
                print(f"WARNING - Imperfect solution used for [{pokemon_name}] - {fit_result_to_use_for_mon}")

            solved_data.append(fit_result_to_use_for_mon | {"pokemon": pokemon_name})
        except IndexError:
            print(f"WARNING - No solution found for [{pokemon_name}], default is used")
            solved_data.append({
                "pokemon": pokemon_name,
                "ing": 0.2,
                "skl": 0.02,
                "result": RpFitResult.FAILED
            })

    print(f"{"=" * 25} Final Results {"=" * 25}")
    for solution in solved_data:
        print(
            f"{solution["pokemon"]:>25} - "
            f"[Ing] {solution["ing"]:6.2%} [Skl] {solution["skl"]:6.2%} ({solution["result"].name})"
        )

    return solved_data


def is_all_last_fit_perfect(
    data: DataFrame,
    last_fit: dict[str, LastFitData],
    x0,
    unpack_info
) -> bool:
    for pokemon_name, grouped in data.groupby("Pokemon"):
        pokemon_name: str

        if not is_pokemon_included_for_rp_model(pokemon_name):
            continue

        last_fit_of_pokemon = last_fit.get(pokemon_name, LastFitData(ing=0.2, skl=0.02))

        computed = make_precomputed_columns(grouped)
        reference_rp = grouped["RP"].astype(float64).to_numpy()

        rate_combo, fit_result = get_rate_combo_fit_result(
            pokemon_name,
            None,
            last_fit_of_pokemon,
            reference_rp,
            x0,
            unpack_info,
            grouped,
            computed,
            initiator="Validate"
        )

        if fit_result != RpFitResult.PERFECT:
            print(f"Last fit for Pokemon is not perfect - {pokemon_name}: {rate_combo}")
            return False

    return True
