from typing import Generator

from .const import MAX_FIT_RADIUS
from ..const import ING_MAX, ING_MIN, SKILL_MAX, SKILL_MIN, TICK_INTERVAL
from ...type import LastFitData


def traverse_at_radius_bfs(center: LastFitData, radius_tick: int, point_gap: float) -> Generator[LastFitData, None, None]:
    radius = radius_tick * point_gap

    for offset_tick in range(-radius_tick + 1, radius_tick + 1):
        offset = offset_tick * point_gap

        yield LastFitData(ing=center.ing + radius, skl=center.skl + offset)
        yield LastFitData(ing=center.ing - radius, skl=center.skl - offset)

    for offset_tick in range(-radius_tick + 1, radius_tick + 1):
        offset = offset_tick * point_gap

        yield LastFitData(ing=center.ing + offset, skl=center.skl - radius)
        yield LastFitData(ing=center.ing - offset, skl=center.skl + radius)


def traverse_last_fit_bfs(
    center: LastFitData,
    /,
    max_radius: int = MAX_FIT_RADIUS,
    skip_center: bool = False,
    point_gap: float = TICK_INTERVAL,
) -> Generator[LastFitData, None, None]:
    def traverser() -> Generator[LastFitData, None, None]:
        if not skip_center:
            yield center

        for radius_tick in range(1, max_radius):
            yield from traverse_at_radius_bfs(center, radius_tick, point_gap)

    for fit in traverser():
        if ING_MIN <= fit.ing <= ING_MAX and SKILL_MIN <= fit.skl <= SKILL_MAX:
            yield fit
