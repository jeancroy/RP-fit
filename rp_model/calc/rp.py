from types import SimpleNamespace

import numpy as np

from .fit_options import FitOptions
from .game import game
from ..utils import floor, optional_floor, unpack


class RPModelData:
    def __init__(self, data, computed, variables):
        self.data = data
        self.computed = computed
        self.vars = variables


def total_ing_value(model):
    ing1_amount = 1.0 + model.computed.has_class["Ingredients"]
    ing2_amount = model.computed.ing2_amount

    ing1_power = model.computed.ing1_power_base
    ing2_power = model.computed.ing2_power_base

    # Equal weigh of 1 has been confirmed
    ing1_weigh = 1.0
    ing2_weigh = (model.computed.ing2_amount > 0)

    # weighted average of the N ingredients and their amount
    ing_value = ing1_amount * ing1_power
    ing_value += ing2_amount * ing2_power
    ing_value /= (ing1_weigh + ing2_weigh)

    # Add the growth curve
    ing_value *= ing_growth(model)

    return ing_value


def ing_growth(model):
    a = model.vars["Ing Growth Poly"]
    x = model.data["Level"].to_numpy()
    g = ((a[0] * x) + a[1]) * x + a[2]
    return 1.0 + g


def ber_amount(model):
    return 1.0 + model.computed.has_class["Berries"] + model.computed.has_subskill["Berry Finding S"]


def ber_value_at_level(model):
    return model.computed.berry_power_at_level


def ing_fraction(model):
    return model.vars["Pokemons ing fractions"][model.computed.ing_positions]


def skl_product(model):
    return model.vars["Pokemons skill products"][model.computed.skl_positions]


def skl_growth(model):
    a = np.array(list(map(lambda x: model.vars[x][0], model.data["MSkill"])))
    b = np.array(list(map(lambda x: model.vars[x][1], model.data["MSkill"])))

    growth = a * np.exp(b * model.data["MS lvl"].to_numpy())
    growth[model.data["MS lvl"] == 1] = 1.0

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
        bonus = bonus + model.computed.has_subskill[name] * model.vars[name]

    return bonus


def compute_rp(variables, data, computed, unpack_info):
    if computed is None:
        computed = make_precomputed_columns(data)

    unpacked = unpack(variables, unpack_info)
    m = RPModelData(data, computed, unpacked)

    # print("variables", digest(variables))
    # print("unpacked", digest(unpacked))
    # print("calc", digest(m))

    #     return compute_rp_from_model(calc)
    #
    #
    # def compute_rp_from_model(m):

    ing = ing_fraction(m) * ing_modifier(m)

    ingredients_value = ing * total_ing_value(m)
    berries_value = (1.0 - ing) * ber_amount(m) * ber_value_at_level(m)
    main_skill_value = skl_product(m) * skl_modifier(m) * skl_growth(m)

    help_count = fractional_help_count(m)
    energy_correction = energy_modifier(m)
    bonus = bonus_subskill(m)

    # Optional flooring of the three main components in RP.
    # (This does not seem to improve the guess, maybe they are multiplied by help count then floored ?)
    ingredients_value = optional_floor(ingredients_value, FitOptions.rounding.components, 0.01)
    berries_value = optional_floor(berries_value, FitOptions.rounding.components, 0.01)
    main_skill_value = optional_floor(main_skill_value, FitOptions.rounding.components, 0.01)

    # Optional flooring of the bonus together with energy
    rounded_bonus = optional_floor(bonus * energy_correction, FitOptions.rounding.bonus, 0.01)

    # core rp formula
    rp = rounded_bonus * help_count * (ingredients_value + berries_value + main_skill_value)

    # Optional final rounding of RP value
    return rp  # optional_round(rp, FitOptions.rounding.final_rp, 1.0)


# Minimum columns for data:
# ------------------------
# Pokemon
# Nature
# Level
# MS Level
# Sub Skill 1
# Sub Skill 2
# Ing2P
# Amnt
#
# Also required, but we could compute from dex
# ---------------------------------------------
# Class
# Freq1
# Ing1P
# Berry1
# BerryL
def make_precomputed_columns(data):
    computed = SimpleNamespace()

    # All the formula "IF" are implemented as one-hot vector  (0,1)

    # Specialty (Class)

    classes = ["Ingredients", "Berries", "Skills"]
    computed.has_class = dict([(c, (data["Class"] == c).astype(int).to_numpy()) for c in classes])

    # Natures

    natures = game.data.natures
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

    subskills = game.data.subskills
    subs = subskills["Subskill"].unique()

    computed.has_subskill = \
        dict([
            (
                s,
                (
                        ((data["Sub Skill 1"].str.lower() == s.lower()) & (data["Level"] >= 10)) |
                        ((data["Sub Skill 2"].str.lower() == s.lower()) & (data["Level"] >= 25)) |
                        ((data["Sub Skill 3"].str.lower() == s.lower()) & (data["Level"] >= 50))
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
            1  # Speed is the only nature where positive effect is a subtraction
            - (computed.has_positive_trait["Speed of Help"] * game.natures.soh_effect)
            + (computed.has_negative_trait["Speed of Help"] * game.natures.soh_effect)
    )

    subskill_correction = (
            1
            - (computed.has_subskill["Helping Speed S"] * game.subskills.help_s_effect)
            - (computed.has_subskill["Helping Speed M"] * game.subskills.help_m_effect)
    )

    final_period = computed.period_level * nature_correction * subskill_correction

    # We are not sure if we need to floor the time period
    final_period = optional_floor(
        final_period, FitOptions.rounding.period, 0.01)

    # But we are almost certain we need to floor the help per hour
    computed.helps_per_hour = floor(3600 / final_period, 0.01)

    # map data point to a pokemon index

    pokemon_to_position = {}

    pokemons = game.data.pokedex["Pokemon"]
    pokemon_to_position = dict(zip(pokemons, range(len(pokemons))))

    data_pokemon_positions = np.array(list(map(lambda x: pokemon_to_position[x], data["Pokemon"])))
    computed.ing_positions = data_pokemon_positions
    computed.skl_positions = data_pokemon_positions

    return computed
