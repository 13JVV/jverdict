"""합성데이터 생성기: 재현성과 정답 기록이 맞는지 확인한다."""
import numpy as np
import pandas as pd
import pytest

from jverdict.synth.generator import (Effect, GroupSpec, MeasurementBreak, Missing, Outage,
                                      PartialDays, SynthSpec, generate)
from jverdict.synth.scenarios import SCENARIOS

GROUPS = [GroupSpec("일반", 30000, noise_sd=0.03), GroupSpec("청소년", 10000, noise_sd=0.03)]


def _spec(inj, **kw):
    return SynthSpec("2023-01-01", "2024-12-31", GROUPS, injections=inj, **kw)


def test_same_seed_same_data():
    a = generate(_spec([Effect("청소년", "2024-06-01", 0.2)]), seed=7)
    b = generate(_spec([Effect("청소년", "2024-06-01", 0.2)]), seed=7)
    pd.testing.assert_frame_equal(a.df, b.df)
    c = generate(_spec([Effect("청소년", "2024-06-01", 0.2)]), seed=8)
    assert not a.df["value"].equals(c.df["value"])


def test_effect_moves_latent_and_observed():
    base = generate(_spec([]), seed=1).latent
    eff = generate(_spec([Effect("청소년", "2024-06-01", 0.3)]), seed=1).latent
    ratio = (eff.loc["2024-07":, "청소년"].sum() / base.loc["2024-07":, "청소년"].sum())
    assert ratio == pytest.approx(np.exp(0.3), rel=0.03)
    assert eff.loc[:"2024-05", "청소년"].sum() == base.loc[:"2024-05", "청소년"].sum()


def test_measurement_break_changes_record_not_behaviour():
    sd = generate(_spec([MeasurementBreak("청소년", "2024-06-01", keep=0.2, adoption=0)]), seed=2)
    obs = sd.df[sd.df.group == "청소년"].set_index("time")["value"]
    lat = sd.latent["청소년"]
    share = obs.loc["2024-07":].sum() / lat.loc["2024-07":].sum()
    assert share == pytest.approx(0.2, abs=0.01)
    assert obs.loc[:"2024-05-31"].sum() == lat.loc[:"2024-05-31"].sum()


def test_reclass_creates_new_category_from_break_date():
    sd = generate(_spec([MeasurementBreak("청소년", "2024-06-01", keep=0.5, reclass_to="기타")]), seed=3)
    first = sd.df.loc[sd.df.group == "기타", "time"].min()
    assert first == pd.Timestamp("2024-06-01")
    moved = sd.df[sd.df.group == "기타"]["value"].sum()
    lost = sd.latent.loc["2024-06-01":, "청소년"].sum() - sd.df.query("group == '청소년' and time >= '2024-06-01'")["value"].sum()
    assert moved == lost   # reclass_share=1.0이면 빠진 만큼 그대로 옮겨 간다


def test_missing_and_partial_tail_and_truth():
    sd = generate(_spec([Missing("2024-03-01", "2024-03-31", None, partial_tail=3), PartialDays(4)]), seed=4)
    days = set(sd.df.time.dt.strftime("%Y-%m-%d"))
    assert not any(d.startswith("2024-03") for d in days)
    kinds = [i["kind"] for i in sd.truth["injections"]]
    assert kinds.count("missing") == 1 and kinds.count("partial_days") == 1
    miss = next(i for i in sd.truth["injections"] if i["kind"] == "missing")
    assert miss["tail_dates"] == ["2024-04-01", "2024-04-02", "2024-04-03"]
    pdays = next(i for i in sd.truth["injections"] if i["kind"] == "partial_days")["dates"]
    assert len(pdays) == 4


def test_outage_scales_all_groups():
    sd = generate(_spec([Outage("2024-06-01", "2024-06-30", keep=0.3)]), seed=5)
    w = sd.df.pivot_table(index="time", columns="group", values="value", aggfunc="sum")
    share = w.loc["2024-06"].sum() / sd.latent.loc["2024-06"].sum()
    assert share["일반"] == pytest.approx(0.3, abs=0.02) and share["청소년"] == pytest.approx(0.3, abs=0.02)


def test_monthly_units_panel():
    spec = SynthSpec("2019-01-01", "2020-12-01", GROUPS, "MS", units=5, holidays=False)
    sd = generate(spec, seed=6)
    assert sd.df["unit"].nunique() == 5 and sd.df["time"].dt.day.eq(1).all()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_every_scenario_builds(name):
    spec, case = SCENARIOS[name].build(np.random.default_rng(0))
    sd = generate(spec, 0)
    assert len(sd.df) > 0
    assert case.label == SCENARIOS[name].label
