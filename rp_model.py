import numpy as np
from types import SimpleNamespace

from game_model import game
from utils.variables import unpack
    
def get_model(variables, _data, _computed, _unpack_info):

    model = SimpleNamespace()
    model.data = _data
    model.computed = _computed
    model.vars = unpack(variables, _unpack_info)
    return model


def total_ing_value(model):
              
    ing1_amount = 1.0 + model.computed.has_class["Ingredients"]
    ing2_amount = model.computed.ing2_amount
    
    ing1_power = model.computed.ing1_power_base
    ing2_power = model.computed.ing2_power_base
    
    ing1_weigth = 1.0
    ing2_weigth = model.vars["Ing 2 Weigth"] * (model.computed.ing2_amount > 0)

    # weighted average of the N ingredients and their amount
    ing_value  = ing1_weigth * ing1_amount * ing1_power
    ing_value += ing2_weigth * ing2_amount * ing2_power
    ing_value /= ( ing1_weigth + ing2_weigth )
    
    # Add the growth curve
    ing_value *= ing_growth(model)
    
    return ing_value

def ing_growth(model):
    
    a = model.vars["Ing Growth Poly"]
    x = model.data["Level"].to_numpy()
    g = ((a[0] * x) + a[1]) * x + a[2]
    return  ( 1.0 + g )

def ber_amount(model):
    
    return 1.0 + model.computed.has_class["Berries"] + model.computed.has_subskill["Berry Finding S"]

def ber_value_at_level(model):
    
    return model.computed.berry_power_at_level

def ing_fraction(model):
    
    return model.vars["Pokemons ing fractions"][ model.computed.ing_positions ]

def skl_product(model):
    
    return model.vars["Pokemons skill products"][ model.computed.skl_positions ]

def skl_growth(model):
    
    a = np.array(list(map(lambda x: model.vars[x][0], model.data["MSkill"])))
    b = np.array(list(map(lambda x: model.vars[x][1], model.data["MSkill"]))) 
 
    growth = a * np.exp( b * model.data["MS lvl"].to_numpy() )
    growth[ model.data["MS lvl"] == 1] = 1.0
    
    return growth


def ing_modifier(model):
    
    nature_correction = (
        1.0
        + (model.computed.has_positive_trait["Ingredient Finding"] * game.natures.ing_effect) 
        - (model.computed.has_negative_trait["Ingredient Finding"] * game.natures.ing_effect)
        )

    subskill_correction = (
        1.0
        + (model.computed.has_subskill["Ingredient Finder S"] * game.subskills.ing_s_effect) 
        + (model.computed.has_subskill["Ingredient Finder M"] * game.subskills.ing_m_effect)
        )
    
    return nature_correction * subskill_correction


def skl_modifier(model):
       
    nature_correction = (
        1.0
        + (model.computed.has_positive_trait["Main Skill Chance"] * game.natures.msc_effect) 
        - (model.computed.has_negative_trait["Main Skill Chance"] * game.natures.msc_effect)
        )

    subskill_correction = (
        1.0
        + (model.computed.has_subskill["Skill Trigger S"] * game.subskills.trigger_s_effect) 
        + (model.computed.has_subskill["Skill Trigger M"] * game.subskills.trigger_m_effect)
        )
    
    return nature_correction * subskill_correction

def energy_modifier(model):
        
    return (
        1.0
        + (model.computed.has_positive_trait["Energy Recovery"] * game.natures.energy_effect)
        - (model.computed.has_negative_trait["Energy Recovery"] * game.natures.energy_effect)
        )

def fractional_help_count(model):
     return 5.0 * model.computed.helps_per_hour

def bonus_subskill(model):
    
    bonus = 1.0
    
    for name in game.subskills.bonus_names:
        bonus = bonus + model.computed.has_subskill[ name ] * model.vars[ name ]
    
    return bonus


def compute_rp(variables, _data, _computed, _unpack_info):
    
    m = get_model(variables, _data, _computed, _unpack_info)
    
    ing = ing_fraction(m) * ing_modifier(m)
    
    ingredients_value = ing * total_ing_value(m)
    berries_value =  (1.0-ing) * ber_amount(m) * ber_value_at_level(m)
    mainskill_value = skl_product(m) * skl_modifier(m) * skl_growth(m)
    
    help_count = fractional_help_count(m)   
    energy_correction = energy_modifier(m)
    bonus = bonus_subskill(m)
        
    rp = bonus * help_count * energy_correction * (ingredients_value + berries_value + mainskill_value)
    
    return rp


def make_precomputed_columns(data):

    computed = SimpleNamespace()

    # All the formula "IF" are implemented as one-hot vector  (0,1)

    # Specialty (Class)

    classes = ["Ingredients", "Berries", "Skills"]
    computed.has_class = dict([(c, (data["Class"] == c).astype(int).to_numpy()) for c in classes])

    # Natures

    natures = game.natures.data
    traits = natures["TraitPos"].unique()

    natures_with_positive_trait = \
        dict([(t, natures[natures["TraitPos"] == t]["Nature"].tolist()) for t in traits])

    natures_with_negative_trait = \
        dict([(t, natures[natures["TraitNeg"] == t]["Nature"].tolist()) for t in traits])

    computed.has_positive_trait = \
        dict([(t, data["Nature"].isin(natures_with_positive_trait[t]).astype(int).to_numpy()) for t in traits])

    computed.has_negative_trait = \
        dict([(t, data["Nature"].isin(natures_with_negative_trait[t]).astype(int).to_numpy()) for t in traits])

    # Sub-skills

    subskills = game.subskills.data
    subs = subskills["Subskill"].unique()

    computed.has_subskill = dict([(s,
                                   (((data["Sub Skill 1"] == s) & (data["Level"] >= 10)) |
                                    ((data["Sub Skill 2"] == s) & (data["Level"] >= 25))
                                    ).astype(int).to_numpy()
                                   )
                                  for s in subs
                                  ])

    # Food
    # We could redo that work, but it's not related to the optimization
    computed.ing1_power_base = data["Ing1P"].to_numpy()
    computed.berry_power_base = data["Berry1"].to_numpy()
    computed.berry_power_at_level = data["BerryL"].to_numpy()

    # Ing2
    computed.ing2_power_base = data["Ing2P"].to_numpy()
    computed.ing2_amount = data["Amnt"].to_numpy()

    # Here, we will reproduce the Help/hr information as a test of using those one-hot vectors.
    computed.period_base = data["Freq1"].to_numpy()
    computed.period_level = computed.period_base * ((501 - data["Level"].to_numpy()) / 500.0)

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

    computed.helps_per_hour = np.floor(100 * 3600 / data_period_level_nature_subskill) / 100

    # map data point to a pokemon index

    pokemon_to_position = {}

    pokemons = game.pokedex.data["Pokemon"]
    pokemon_to_position = dict(zip(pokemons, range(len(pokemons))))

    data_pokemon_positions = np.array(list(map(lambda x: pokemon_to_position[x], data["Pokemon"])))
    computed.ing_positions = data_pokemon_positions
    computed.skl_positions = data_pokemon_positions

    return computed