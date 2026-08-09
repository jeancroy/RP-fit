from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
from pandas import DataFrame

from ..calc.rp import (
    RPModelData,
    bonus_subskill,
    energy_modifier,
    final_berries_value,
    final_ingredients_value,
    fractional_help_count,
    ing_modifier,
    skill_value,
    skl_modifier,
)
from ..utils import truncate


def get_rp_rate_components(
    data: DataFrame,
    computed: SimpleNamespace,
    ingredient_rates: npt.NDArray[np.float64],
    skill_rates: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    model = RPModelData(data, computed, {})
    help_count = fractional_help_count(model)[np.newaxis, :]

    ingredient_ratio = truncate(ingredient_rates[:, np.newaxis] * ing_modifier(model), 4)
    ingredient_values = (
        truncate(help_count * ingredient_ratio * final_ingredients_value(model), 2)
        + truncate(help_count * (1.0 - ingredient_ratio) * final_berries_value(model), 2)
    )

    skill_ratio = truncate(skill_rates[:, np.newaxis] * skl_modifier(model), 4)
    skill_values = truncate(
        help_count * (skill_ratio + np.finfo(np.float64).eps) * skill_value(model),
        2,
    )
    bonus_multipliers = truncate(bonus_subskill(model) * energy_modifier(model), 2)
    return ingredient_values, skill_values, bonus_multipliers
