from numpy import float64
from pandas import DataFrame

from .fit import get_rate_combo_fit_result
from ..calc import make_precomputed_columns
from ..enum import RpFitResult
from ..env import is_pokemon_included_for_rp_model
from ..type import LastFitData


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

        last_fit_of_pokemon = last_fit.get(pokemon_name, LastFitData.default())

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
            initiator="Validate",
            print_func=print,
        )

        if fit_result != RpFitResult.PERFECT:
            print(f"Last fit for Pokemon is not perfect - {pokemon_name}: {rate_combo}")
            return False

    return True
