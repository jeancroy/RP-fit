import numpy as np
import numpy.typing as npt

from ..const import ING_MAX, ING_MIN, SKILL_MAX, SKILL_MIN, TICK_INTERVAL


def _get_rate_ticks(minimum: float, maximum: float) -> npt.NDArray[np.float64]:
    minimum_tick = round(minimum / TICK_INTERVAL)
    maximum_tick = round(maximum / TICK_INTERVAL)
    return np.arange(minimum_tick, maximum_tick + 1, dtype=np.float64) * TICK_INTERVAL


def get_ingredient_rate_ticks() -> npt.NDArray[np.float64]:
    return _get_rate_ticks(ING_MIN, ING_MAX)


def get_skill_rate_ticks() -> npt.NDArray[np.float64]:
    return _get_rate_ticks(SKILL_MIN, SKILL_MAX)
