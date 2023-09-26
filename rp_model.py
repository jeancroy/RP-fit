import numpy as np
import pandas as pd
from types import SimpleNamespace 

from game_model import game
from variables import unpack
    
def get_model(variables, _data, _computed, _unpack_info):

    model = SimpleNamespace()
    model.data = _data
    model.computed = _computed
    model.vars = unpack(variables, _unpack_info)
    return model

def ing1_value_at_level(model):
    
    a = model.vars["Ing Growth Poly"]
    x = model.data["Level"].to_numpy()
    g = ((a[0] * x) + a[1]) * x + a[2]

    return model.computed.ing1_power_base * ( 1.0 + g )

def ing1_amount(model):
    return 1.0 + model.computed.has_class["Ingredients"]

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


def computeRP(variables, _data, _computed, _unpack_info):
    
    m = get_model(variables, _data, _computed, _unpack_info)
    
    ing = ing_fraction(m) * ing_modifier(m)  
    
    ingredients_value = ing * ing1_amount(m) * ing1_value_at_level(m)

    berries_value =  (1.0-ing) * ber_amount(m) * ber_value_at_level(m)
    
    mainskill_value = skl_product(m) * skl_modifier(m) * skl_growth(m)
    
    help_count = fractional_help_count(m)
    
    energy_correction = energy_modifier(m)
    
    bonus = bonus_subskill(m)
        
    rp = bonus * help_count * energy_correction * (ingredients_value + berries_value + mainskill_value)
    
    return rp