import numpy as np
from pandas import DataFrame

from .fit import get_rp_diff
from .typedef import OptimizerFitContext
from ..calc import make_precomputed_columns
from ..type import LastFitData
from ..utils.thread_safe_print import thread_safe_print


def is_last_fit_exact(
    pokemon_name: str,
    data_of_pokemon: DataFrame,
    last_fit: LastFitData | None,
    x0,
    unpack_info,
) -> bool:
    last_fit_of_pokemon = last_fit or LastFitData.default()

    computed = make_precomputed_columns(data_of_pokemon)
    reference_rp = data_of_pokemon["RP"].to_numpy(dtype=np.float64)

    context = OptimizerFitContext(
        pokemon_name=pokemon_name,
        pokemon_data_of_group=data_of_pokemon,
        x0=x0,
        unpack_info=unpack_info,
        print_func=thread_safe_print,
    )
    if get_rp_diff(context, reference_rp, computed, last_fit_of_pokemon).any():
        thread_safe_print(f"Last fit for Pokemon is not exact - {pokemon_name}: {last_fit_of_pokemon}")
        return False

    return True
