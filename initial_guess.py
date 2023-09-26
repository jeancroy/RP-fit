from game_model import *

def make_initial_guess():

    # Here we build the initial guess

    initial = { }

    # Load Initial guess for the ing% and skillProduct ( skill% * skillValue ) from pokedex 

    previous_ing_fractions = []
    previous_skl_products = []

    for record in game.pokedex.data.to_dict(orient='records') :   

        # Last fit
        previous_ing_fractions.append(record["Last fit ing"])
        previous_skl_products.append(record["Last fit skl"])
    
    initial["Pokemons ing fractions"] = previous_ing_fractions
    initial["Pokemons skill products"] = previous_skl_products

    # Intial guess for skill growth
    # We assume that the conversion from level 1 to level L
    # Has the shape a*exp(b*L). Initial guess for a,b fitted on charge strength

    for record in game.mainskills.data.to_dict(orient='records') :   
        skillname = record["Skill"]
        initial[ skillname ] = [0.7462, 0.3224]

    # Initial guess for ingredient growth.
    # Comes from a previous fit
    # Numpy poly convention is highest degree first

    initial["Ing Growth Poly"] = [0.00018948, 0.00306669, -0.00173611] 

    # Add our guess for the sub skills that multiply the whole rp

    subskills = game.subskills.data
    bonus = subskills[ subskills["Subskill"].isin(game.subskills.bonus_names) ]

    for record in bonus.to_dict(orient='records') :
        initial[record["Subskill"]] = record["RP Bonus Estimate"]

    return initial