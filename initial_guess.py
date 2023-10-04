from game_model import game
import numpy as np


def make_initial_guess():
    # Here we build the initial guess

    initial = {}

    # Load Initial guess for the ing% and skillProduct ( skill% * skillValue ) from Pokedex

    previous_ing_fractions = []
    previous_skl_products = []

    for record in game.data.pokedex.to_dict(orient='records'):
        # Last fit
        previous_ing_fractions.append(record["Last fit ing"])
        previous_skl_products.append(record["Last fit skl"])

    initial["Pokemons ing fractions"] = np.array(previous_ing_fractions)
    initial["Pokemons skill products"] = np.array(previous_skl_products)

    # Initial guess for skill growth,
    # We assume that the conversion from level 1 to level L
    # Has the shape a*exp(b*L). Initial guess for a,b fitted on charge strength

    for record in game.data.mainskills.to_dict(orient='records'):
        skill_name = record["Skill"]
        initial[skill_name] = np.array([0.7462, 0.3224])

    # Initial guess for ingredient growth.
    # Comes from a previous fit
    # Numpy poly convention is the highest degree first

    initial["Ing Growth Poly"] = np.array( [0.00018848, 0.00314556, 0.0033453] )

    # Add our guess for the sub-skills that multiply the whole rp

    subskills = game.data.subskills
    bonus = subskills[subskills["Subskill"].isin(game.subskills.bonus_names)]

    for record in bonus.to_dict(orient='records'):
        initial[record["Subskill"]] = record["RP Bonus Estimate"]

    return initial
