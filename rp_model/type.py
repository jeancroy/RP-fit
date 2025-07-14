from typing import NamedTuple


class LastFitData(NamedTuple):
    ing: float
    skl: float

    @staticmethod
    def default():
        return LastFitData(ing=0.2, skl=0.02)
