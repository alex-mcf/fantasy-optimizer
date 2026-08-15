from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class LeagueConfig:
    """Roster settings that affect draft rounds and replacement levels."""

    league_size: int = 12
    qb: int = 1
    rb: int = 2
    wr: int = 2
    te: int = 1
    flex: int = 1
    superflex: int = 0

    def __post_init__(self) -> None:
        slots = (self.qb, self.rb, self.wr, self.te, self.flex, self.superflex)
        if self.league_size <= 0 or any(slot < 0 for slot in slots):
            raise ValueError(
                "League size must be positive and roster slots cannot be negative."
            )

    def replacement_ranks(self) -> dict[str, int]:
        """Estimate replacement ranks from starters and flexible roster slots.

        FLEX demand is allocated 40% to RB, 50% to WR, and 10% to TE.
        Superflex slots are treated as QB demand. These assumptions are explicit
        so draft-behavior estimates can replace them later.
        """
        flex_demand = self.league_size * self.flex
        return {
            "QB": max(1, self.league_size * (self.qb + self.superflex)),
            "RB": max(1, self.league_size * self.rb + ceil(flex_demand * 0.40)),
            "WR": max(1, self.league_size * self.wr + ceil(flex_demand * 0.50)),
            "TE": max(1, self.league_size * self.te + ceil(flex_demand * 0.10)),
        }


DEFAULT_LEAGUE_CONFIG = LeagueConfig()

# Backward-compatible dictionary for callers that used the original module.
league_settings = {
    "league_size": DEFAULT_LEAGUE_CONFIG.league_size,
    "positions": {
        "QB": DEFAULT_LEAGUE_CONFIG.qb,
        "RB": DEFAULT_LEAGUE_CONFIG.rb,
        "WR": DEFAULT_LEAGUE_CONFIG.wr,
        "TE": DEFAULT_LEAGUE_CONFIG.te,
        "FLEX": DEFAULT_LEAGUE_CONFIG.flex,
        "SuperFLEX": DEFAULT_LEAGUE_CONFIG.superflex,
    },
}
