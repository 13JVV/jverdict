"""합성데이터: 정답(효과·함정)을 아는 데이터로 툴의 정확도를 잰다."""
from .generator import (Effect, GroupSpec, MeasurementBreak, Missing, Outage, PartialDays, PreTrend,
                        Shock, SynthData, SynthSpec, generate)
from .scenarios import SCENARIOS

__all__ = ["generate", "SynthSpec", "SynthData", "GroupSpec", "Effect", "PreTrend", "Shock",
           "MeasurementBreak", "Outage", "Missing", "PartialDays", "SCENARIOS"]
