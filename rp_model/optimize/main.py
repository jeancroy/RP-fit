from numpy import float64
from pandas import DataFrame

from .check import is_rate_combo_fit
from .traverse import traverse_last_fit
from ..calc import make_precomputed_columns
from ..type import LastFitData


# x0 is a mystery array of numbers that would create different % combinations with `unpack_info`
# x0 + unpack_info = a different combination of rates
def run_optimizer(
    data: DataFrame,
    last_fit: dict[str, LastFitData],
    x0,
    unpack_info
):
    solved_data = []

    for pokemon_name, grouped in data.groupby("Pokemon"):
        pokemon_name: str

        last_fit_of_pokemon = last_fit.get(pokemon_name, LastFitData(ing=0.2, skl=0.02))

        computed = make_precomputed_columns(grouped)
        reference_rp = grouped["RP"].astype(float64).to_numpy()

        is_pokemon_solved = False
        for idx, rate_combo in enumerate(traverse_last_fit(last_fit_of_pokemon)):
            if not is_rate_combo_fit(
                    pokemon_name,
                    idx,
                    rate_combo,
                    reference_rp,
                    x0,
                    unpack_info,
                    grouped,
                    computed,
                    initiator="Solve"
            ):
                continue

            solved_data.append({"pokemon": pokemon_name, "ing": rate_combo.ing, "skl": rate_combo.skl})
            is_pokemon_solved = True
            break

        if not is_pokemon_solved:
            print(f"WARNING - No solution found for [{pokemon_name}], last fit ({last_fit_of_pokemon}) is used")
            solved_data.append({
                "pokemon": pokemon_name,
                "ing": last_fit_of_pokemon.ing,
                "skl": last_fit_of_pokemon.skl
            })

    return solved_data


def is_all_last_fit_valid(
    data: DataFrame,
    last_fit: dict[str, LastFitData],
    x0,
    unpack_info
) -> bool:
    for pokemon_name, grouped in data.groupby("Pokemon"):
        pokemon_name: str

        last_fit_of_pokemon = last_fit.get(pokemon_name, LastFitData(ing=0.2, skl=0.02))

        computed = make_precomputed_columns(grouped)
        reference_rp = grouped["RP"].astype(float64).to_numpy()

        if not is_rate_combo_fit(
                pokemon_name,
                None,
                last_fit_of_pokemon,
                reference_rp,
                x0,
                unpack_info,
                grouped,
                computed,
                initiator="Validate"
        ):
            print(f"Last fit validation failed for Pokemon {pokemon_name}: {last_fit_of_pokemon}")
            return False

    return True
