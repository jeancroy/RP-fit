from types import SimpleNamespace

import numpy as np

from .game import game
from ..utils import unpack, truncate, lookup_table


class RPModelData:
    def __init__(self, data, computed, variables):
        self.data = data
        self.computed = computed
        self.vars = variables


def compute_rp_components(variables, data, computed, unpack_info):
    if computed is None:
        computed = make_precomputed_columns(data)

    unpacked = unpack(variables, unpack_info)
    m = RPModelData(data, computed, unpacked)

    skill_ratio = truncate(skl_chance(m) * skl_modifier(m), 4)
    ingredient_ratio = truncate(ing_fraction(m) * ing_modifier(m), 4)
    berry_ratio = (1.0 - ingredient_ratio)

    help_count = fractional_help_count(m)
    ingredients_value = truncate(help_count * ingredient_ratio * final_ingredients_value(m), 2)
    berries_value = truncate(help_count * berry_ratio * final_berries_value(m), 2)
    main_skill_value = truncate(help_count * skill_ratio * skill_value(m), 2)

    # Optional flooring of the bonus together with energy
    bonus_multipliers = truncate(bonus_subskill(m) * energy_modifier(m), 2)

    # Put items together.
    rp = np.round(bonus_multipliers * (ingredients_value + berries_value + main_skill_value))

    return dict({
        "rp_model": rp,
        "ingredients_value_model": ingredients_value,
        "berries_value_model": berries_value,
        "main_skill_value_model": main_skill_value,
        "bonus_multipliers_model": bonus_multipliers,
    })


def compute_rp(variables, data, computed, unpack_info):
    results = compute_rp_components(variables, data, computed, unpack_info)
    return results["rp_model"]


def final_ingredients_value(model):
    ing1_amount = 1.0 + model.computed.has_class["Ingredients"]
    ing2_amount = model.computed.ing2_amount.astype(int)

    ing1_power = model.computed.ing1_power_base
    ing2_power = model.computed.ing2_power_base

    # Equal weigh of 1 has been confirmed
    ing1_weigh = 1.0
    ing2_weigh = (model.computed.ing2_amount.astype(int) > 0)

    # weighted average of the N ingredients and their amount
    ing_value = ing1_amount * ing1_power
    ing_value += ing2_amount * ing2_power
    ing_value /= (ing1_weigh + ing2_weigh)

    # flooring
    ing_value = np.floor(ing_value)

    # scale by the growth curve
    ing_value *= ing_growth(model)

    return ing_value


def ing_growth(model):
    return (
            1.0 + 0.01 * lookup_table(
        model.data["Level"],
        game.data.ing_growth,
        "Level",
        "Ing Growth"
    ).to_numpy()
    )


def final_berries_value(model):
    return ber_amount(model) * ber_value_at_level(model)


def ber_amount(model):
    return 1.0 + model.computed.has_class["Berries"] + model.computed.has_subskill["Berry Finding S"]


def ber_value_at_level(model):
    return model.computed.berry_power_at_level


def ing_fraction(model):
    # 0.abc
    return truncate(model.vars["Pokemons ing fractions"][model.computed.ing_positions], 3)


def skl_chance(model):
    return truncate(model.vars["Pokemons skill chances"][model.computed.skl_positions], 3)


def skill_value(model):
    return model.computed.skill_value_at_level


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

    lookup = game.data.subskills.set_index("Subskill")["RP Bonus Estimate"]

    for name in game.subskills.bonus_names:
        bonus = bonus + model.computed.has_subskill[name] * lookup[name]

    return bonus


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
#
# Class
# Freq1
# Ing1P
# Berry1
# BerryL
# SklVal
# Helps per hour
#
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

    # Main skill and level
    computed.skill_value_at_level = data["SklVal"].to_numpy()

    computed.period_base = data["Freq1"].to_numpy()
    computed.helps_per_hour = data["Helps per hour"].to_numpy()

    # We copy help per hour from datasheet during debug, eventually we go back to our own.
    #
    # nature_correction = (
    #         1  # Speed is the only nature where positive effect is a subtraction
    #         - (computed.has_positive_trait["Speed of Help"] * game.natures.soh_effect)
    #         + (computed.has_negative_trait["Speed of Help"] * game.natures.soh_effect)
    # )
    #
    # subskill_correction = (
    #         1
    #         - (computed.has_subskill["Helping Speed S"] * game.subskills.help_s_effect)
    #         - (computed.has_subskill["Helping Speed M"] * game.subskills.help_m_effect)
    # )
    #
    # level_adjust = ((501 - data["Level"].to_numpy()) / 500.0)
    # final_correction = truncate(level_adjust * nature_correction * subskill_correction, 4)
    # final_period = computed.period_base * final_correction
    #
    # # But we are almost certain we need to floor the help per hour
    # computed.helps_per_hour = truncate(3600 / final_period, 2)
    #

    # map data points to a pokemon index

    pokemons = game.data.pokedex["Pokemon"]
    pokemon_name_to_pokedex_position = dict(zip(pokemons, range(len(pokemons))))

    data_line_to_pokedex_position = np.array(list(map(lambda x: pokemon_name_to_pokedex_position[x], data["Pokemon"])))
    computed.ing_positions = data_line_to_pokedex_position
    computed.skl_positions = data_line_to_pokedex_position

    return computed
