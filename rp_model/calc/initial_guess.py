import numpy as np

from .game import game
from ..utils import RangeInfo


def make_initial_guess():
    # Here we build the initial guess

    initial_guess = {}
    range_info = {}

    # Load Initial guess for the ing% and skillProduct ( skill% * skillValue ) from Pokedex

    previous_ing_fractions = []
    previous_skl_products = []

    for record in game.data.pokedex.to_dict(orient='records'):
        # Last fit
        previous_ing_fractions.append(record["Last fit ing"])
        previous_skl_products.append(record["Last fit skl"])

    initial_guess["Pokemons ing fractions"] = np.array(previous_ing_fractions)
    initial_guess["Pokemons skill products"] = np.array(previous_skl_products)

    range_info["Pokemons ing fractions"] = RangeInfo(0.1, 0.3)
    range_info["Pokemons skill products"] = RangeInfo(5.0, 70.0)

    # Initial guess for skill growth,
    # We assume that the conversion from level 1 to level L
    # Has the shape a*exp(b*L). Initial guess for a,b fitted on charge strength

    for record in game.data.mainskills.to_dict(orient='records'):
        skill_name = record["Skill"]
        initial_guess[skill_name] = np.array([0.7462, 0.3224])
        range_info[skill_name] = RangeInfo(0.2, 0.9)

    # Initial guess for ingredient growth.
    # Comes from a previous fit
    # Numpy poly convention is the highest degree first

    initial_guess["Ing Growth Poly"] = np.array([0.00018848, 0.00314556, 0.0033453])
    range_info["Ing Growth Poly"] = RangeInfo(-0.005, 0.005)

    # Add our guess for the subskills that multiply the whole rp

    subskills = game.data.subskills
    bonus = subskills[subskills["Subskill"].isin(game.subskills.bonus_names)]

    for record in bonus.to_dict(orient='records'):
        initial_guess[record["Subskill"]] = record["RP Bonus Estimate"]
        range_info[record["Subskill"]] = RangeInfo(0.05, 0.30)

    return initial_guess, range_info
