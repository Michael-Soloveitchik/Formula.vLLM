"""Tests for measure.py pure math + results/*.json."""
import json
import math
import os

import measure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- robust_sd -------------------------------------------------------------

def test_robust_sd_known():
    # median=3, abs devs=[2,1,0,1,2], MAD=1 -> 1.4826
    assert measure.robust_sd([1, 2, 3, 4, 5]) == 1.4826
    assert math.isclose(measure.robust_sd([5, 5, 5, 5]), 0.0)
    assert measure.robust_sd([42]) == 0.0          # <2 samples -> 0
    # explicit center overrides the median
    assert math.isclose(measure.robust_sd([1, 2, 3, 4, 5], center=0.0),
                        1.4826 * 3)


# ---- advantage -------------------------------------------------------------

def test_advantage_edges():
    # agent == expert -> 1.0
    assert measure.advantage(10.0, 2.0, 2.0) == 1.0
    # agent == base -> 0.0
    assert measure.advantage(10.0, 10.0, 2.0) == 0.0
    # agent faster than expert -> clipped to 1.0
    assert measure.advantage(10.0, 1.0, 2.0) == 1.0
    # halfway between base and expert -> 0.5
    assert math.isclose(measure.advantage(10.0, 6.0, 2.0), 0.5)
    # degenerate denom (base <= expert) -> 0.0
    assert measure.advantage(2.0, 1.0, 5.0) == 0.0


# ---- Comparison via tiny deterministic fns (no real timing math) -----------

def test_comparison_dataclass_fields():
    c = measure.Comparison(name="x", samples=[0.0, 0.0, 0.0], s_repo=0.0)
    assert c.median_log == 0.0
    assert math.isclose(c.median_pct, 0.0)
    assert c.sigma == 0.0
    assert c.theta == measure.THETA_MIN      # clipped up to floor
    assert c.counts is False and c.broken is False


# ---- results JSON ----------------------------------------------------------

def test_result_21222():
    r = json.load(open(os.path.join(ROOT, "results", "21222.json")))
    assert r["pr"] == 21222
    arms = r["arms"]
    for a in ("base", "gold", "claude", "codex"):
        assert arms[a]["tests"] in ("PASS", "FAIL")
    assert arms["base"]["tests"] == "FAIL"
    assert arms["gold"]["tests"] == "PASS"
    assert r["improved"] is True
    lat = r["latency"]
    assert lat["arms"]["gold"]["mean"] < lat["base"]["mean"]   # gold faster


def test_result_all():
    data = json.load(open(os.path.join(ROOT, "results", "all.json")))
    assert isinstance(data, list)
    assert any(e["pr"] == 21222 for e in data)
