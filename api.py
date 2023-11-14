from model.fit_options import fit_options
from model.initial_guess import make_initial_guess
from model.rp import compute_rp, make_precomputed_columns
from model.data import download_data, refresh_pokedex
from model.game import game

from utils.options import merge_dict_like
from utils.files import save, try_load
from utils.variables import pack, unpack, simplify_opt_result
from utils.hash import digest

import scipy
import os.path as path
import pandas as pd


def update_fit_cached(options=None):
    merge_dict_like(fit_options, options)

    refresh_pokedex()
    data = download_data()
    x0, unpack_info = pack(*make_initial_guess())

    hashid = digest(data, x0)
    filepath = fit_options.get_result_file(hashid)
    opt = try_load(filepath)

    if (opt is None):
        opt = run_optimizer(data, x0, unpack_info)
        save(filepath, simplify_opt_result(opt))

    else:
        print("Loaded from cache")

    sol = unpack(opt.x, unpack_info)

    result = pd.DataFrame({

        "pokemon": game.data.pokedex["Pokemon"],
        "pokemonId": game.data.pokedex["Pokemon ID"],
        "ingredientSplit": sol["Pokemons ing fractions"],
        "skillValue": sol["Pokemons skill products"]

        # todo
        # confidence: A python dict with the keys of ingredient and skill. Sample value: 0.1022 (conf (xxx)*) in the bootstrap one
        # skillPercent: Only needed if there's a field for the model to consume skill value and calculate skill %. Sample value: 6.512

    })

    # Merge with result count

    result = result.set_index("pokemon")
    result['dataCount'] = data.groupby(['Pokemon']).size()

    # If there's no count, Panda uses NaN, and therefore casts everything to double.
    # So we now undo that.
    result['dataCount'] = result['dataCount'].fillna(0).astype(int)

    return result


def run_optimizer(data, x0, unpack_info):
    computed = make_precomputed_columns(data)
    reference_rp = data["RP"].to_numpy()

    def residual(x):
        return reference_rp - compute_rp(x, data, computed, unpack_info)

    opt = scipy.optimize.least_squares(residual, x0, **fit_options.least_squares_kwargs)

    return opt


def path_from_this_file(p):
    return path.join(path.dirname(__file__), p)


def main():
    from utils.display import table

    overwrites = dict(
        result_file_pattern=path_from_this_file("./test/alternative-save-location/some-filename-{hash_id}.pickle"),
        rp_file_id="1kBrPl0pdAO8gjOf_NrTgAPseFtqQA27fdfEbMBBeAhs",
    )

    sol = update_fit_cached(options=overwrites)
    table(sol)


if __name__ == "__main__":
    main()
