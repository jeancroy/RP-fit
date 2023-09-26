import numpy as np
import pandas as pd
from types import SimpleNamespace 
from game_model import game
    
def computedColumns(data):
        
    computed = SimpleNamespace()

    # All the formula "ifs" will be implemented as one-hot vector  (0,1)

    # Specialty (Class)

    classes = ["Ingredients","Berries","Skills"]
    computed.has_class = dict( [(c, (data["Class"] == c).astype(int).to_numpy() ) for c in classes])

    # Natures

    natures = game.natures.data
    traits = natures["TraitPos"].unique()

    natures_with_positive_trait = \
        dict( [( t, natures[ natures["TraitPos"] == t ]["Nature"].tolist() ) for t in traits ] )

    natures_with_negative_trait = \
        dict( [( t, natures[ natures["TraitNeg"] == t ]["Nature"].tolist() ) for t in traits ] )

    computed.has_positive_trait = \
        dict( [(t, data["Nature"].isin(natures_with_positive_trait[t]).astype(int).to_numpy() ) for t in traits] )

    computed.has_negative_trait = \
        dict( [(t, data["Nature"].isin(natures_with_negative_trait[t]).astype(int).to_numpy() ) for t in traits] )

    # Subskills

    subskills = game.subskills.data
    subs = subskills["Subskill"].unique()

    computed.has_subskill = dict([ (s, 
                              ( ( (data["Sub Skill 1"] == s) & (data["Level"] >= 10) ) | 
                                ( (data["Sub Skill 2"] == s) & (data["Level"] >= 25) ) 
                              ).astype(int).to_numpy()
                          )
           for s in subs
         ])

    # Food
    # We could redo that work but it's not related to the optimisation
    computed.ing1_power_base      = data["Ing1P"].to_numpy()
    computed.berry_power_base     = data["Berry1"].to_numpy()
    computed.berry_power_at_level = data["BerryL"].to_numpy()
    
    #Ing2
    computed.ing2_power_base = data["Ing2P"].to_numpy()
    computed.ing2_amount     = data["Amnt"].to_numpy()
    computed.has_ing2        = (data["Level"] >= 30).astype(int).to_numpy()
    
    # otimize ing2weigth using
    #( ing1amount * computed.ing1_power_base + ing2amount*computed.ing2_power_base) / (1 + computed.has_ing2 * ing2weigth)
    
    
    # Here we will reproduce the Help/hr information as a test of using those one-hot vectors.
    computed.period_base  = data["Freq1"].to_numpy()
    computed.period_level = computed.period_base * ((501-data["Level"].to_numpy())/500.0) 

    nature_correction = (
        1  # Speed is the only nature where positive effect is a substraction
        - (computed.has_positive_trait["Speed of Help"] * game.natures.soh_effect)
        + (computed.has_negative_trait["Speed of Help"] * game.natures.soh_effect)
        )

    subskill_correction = (
        1 
        - (computed.has_subskill["Helping Speed S"] * game.subskills.help_s_effect) 
        - (computed.has_subskill["Helping Speed M"] * game.subskills.help_m_effect)
        )

    data_period_level_nature_subskill = computed.period_level * nature_correction * subskill_correction

    computed.helps_per_hour = np.floor(100*3600/data_period_level_nature_subskill)/100
    
    # map data point to a pokemon index
    
    pokemon_to_position = {}
    
    pokemons = game.pokedex.data["Pokemon"]
    pokemon_to_position = dict( zip( pokemons, range( len(pokemons) ) ) )
    
    data_pokemon_positions  = np.array( list( map( lambda x: pokemon_to_position[x], data["Pokemon"]) ) ) 
    computed.ing_positions = data_pokemon_positions
    computed.skl_positions = data_pokemon_positions


    return computed

