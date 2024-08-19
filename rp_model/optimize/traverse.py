import math
from typing import Generator

from ..type import LastFitData

TICK_INTERVAL = 0.001

SKILL_MIN = 0.001
SKILL_MAX = 0.170

ING_MIN = 0.1
ING_MAX = 0.4

MAX_ING_FIT_TICKS = math.ceil((ING_MAX - ING_MIN) / TICK_INTERVAL)

MAX_SKL_FIT_TICKS = math.ceil((SKILL_MAX - SKILL_MIN) / TICK_INTERVAL)

MAX_POSSIBLE_FITS = MAX_ING_FIT_TICKS * MAX_SKL_FIT_TICKS

MAX_FIT_RADIUS = max(MAX_ING_FIT_TICKS, MAX_SKL_FIT_TICKS)


def traverse_at_radius(center: LastFitData, radius_tick: int) -> Generator[LastFitData, None, None]:
    radius = radius_tick * TICK_INTERVAL

    for offset_tick in range(-radius_tick + 1, radius_tick + 1):
        offset = offset_tick * TICK_INTERVAL

        yield LastFitData(ing=center.ing + radius, skl=center.skl + offset)
        yield LastFitData(ing=center.ing - radius, skl=center.skl - offset)

    for offset_tick in range(-radius_tick + 1, radius_tick + 1):
        offset = offset_tick * TICK_INTERVAL

        yield LastFitData(ing=center.ing + offset, skl=center.skl - radius)
        yield LastFitData(ing=center.ing - offset, skl=center.skl + radius)


def traverse_last_fit(
    center: LastFitData,
    /,
    max_radius: int | None = None,
    skip_center: bool = False
) -> Generator[LastFitData, None, None]:
    def traverser() -> Generator[LastFitData, None, None]:
        if not skip_center:
            yield center

        for radius_tick in range(1, max_radius or MAX_FIT_RADIUS):
            yield from traverse_at_radius(center, radius_tick)

    for fit in traverser():
        if ING_MIN <= fit.ing <= ING_MAX and SKILL_MIN <= fit.skl <= SKILL_MAX:
            yield fit
