"""Tests for measure.py pure math + the per-PR results files (data/pr_<n>/results.<hw>.json)."""
import json
import math
import os

from gen import layout
from pipeline import measure

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
    p = layout.results_json_any(21222)
    assert p, "no results file for PR 21222 on any hardware slug"
    r = json.load(open(p))
    assert r["pr"] == 21222
    assert r["hw_slug"] in layout.results_files(21222)
    arms = r["arms"]
    for a in ("base", "gold"):
        assert arms[a]["tests"] in ("PASS", "FAIL", "SKIP")
    assert r["schema"] == "experiments_v2"


def test_result_all():
    data = json.load(open(layout.all_json()))
    assert isinstance(data, list)
    assert any(e["pr"] == 21222 for e in data)
    assert all("hw_slug" in e for e in data)
    man = json.load(open(layout.manifest_json()))
    assert 21222 in man["prs"] and man["files"]["21222"]
