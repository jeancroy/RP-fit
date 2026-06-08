from numpy import float64
from pandas import DataFrame

from .fit import get_rate_combo_fit_result
from .typedef import OptimizerFitContext
from ..calc import make_precomputed_columns
from ..enum import RpFitResult
from ..type import LastFitData
from ..utils.thread_safe_print import thread_safe_print


def is_last_fit_perfect(
    pokemon_name: str,
    data_of_pokemon: DataFrame,
    last_fit: LastFitData | None,
    x0,
    unpack_info,
) -> bool:
    last_fit_of_pokemon = last_fit or LastFitData.default()

    computed = make_precomputed_columns(data_of_pokemon)
    reference_rp = data_of_pokemon["RP"].astype(float64).to_numpy()

    fit_result = get_rate_combo_fit_result(
        OptimizerFitContext(
            pokemon_name=pokemon_name,
            pokemon_data_of_group=data_of_pokemon,
            x0=x0,
            unpack_info=unpack_info,
            print_func=thread_safe_print,
        ),
        last_fit_of_pokemon,
        reference_rp,
        computed,
        initiator="Validate",
    )

    if fit_result.result != RpFitResult.PERFECT:
        thread_safe_print(f"Last fit for Pokemon is not perfect - {pokemon_name}: {fit_result.fit}")
        return False

    return True
