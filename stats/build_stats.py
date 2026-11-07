#!/usr/bin/env python3
"""The numbers behind stats/stats.ipynb and the web Stats pane, written as files from
data/pr_<n>/results.<hw-slug>.json (this machine's slug per PR, else the first slug available).

    python3 stats/build_stats.py              # writes stats/prs.csv, fields.csv, status.csv, gaps.png
    python3 stats/build_stats.py --hw h100-80gb

  prs.csv     one row per measured PR: pr, field, hardware, headline, unit, reported baseline / PR's solution /
              gap, reproduced baseline / PR's solution / sd / gap, gap ratio, verdict, reliability, tests
  fields.csv  per field: n, median / min / max gap ratio, median reported & reproduced gap
  status.csv  every workload PR -> measured | failed | unmet | pending (+ reason), as next_prs.py --status
  gaps.png    2x2: gap ratio per PR, gaps per PR, gap ratio per field, gaps per field
              (#e39b00 reported / #0b8ce8 reproduced; log axes grow from 1x)

gap = PR's solution − baseline, relative to its own baseline so units cancel:
(baseline − PR's solution) ÷ baseline, throughput units flipped. gap ratio = reproduced gap ÷ reported gap
(1 = we see exactly the author's gap). Only real runs count (summary.measured > 0).
run.sh --reproduce calls this after aggregate.py so the files are always current.
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE = "#0b8ce8", "#e39b00"   # reproduced / reported (same as the dashboard)
PR_COLS = ["pr", "field", "hardware", "headline", "unit",
           "reported_baseline", "reported_pr_solution", "reported_marker", "reported_ratio", "reported_gap",
           "reproduced_baseline", "reproduced_pr_solution", "reproduced_ratio", "reproduced_ratio_sd", "reproduced_gap",
           "gap_ratio", "verdict", "reliability", "tests", "title"]


def _up(unit):
    return bool(pd.Series([unit or ""]).str.contains(r"/s$|tok/s|req/s").iloc[0])


def _gap(ratio, up):
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return np.nan
    return ratio - 1 if up else 1 - 1 / ratio


def load_prs(hw=None):
    """DataFrame, one row per PR measured here (summary.measured > 0), headline experiment."""
    rows = []
    for pr in layout.prs():
        p = layout.results_json_any(pr, hw)
        if not p:
            continue
        d = json.load(open(p))
        s = d.get("summary") or {}
        if not s.get("measured"):
            continue
        h = s.get("headline_derived") or {}
        rp, rr = h.get("reproduced") or {}, h.get("reported") or {}
        up = _up(h.get("unit"))
        rep_ratio, meas_ratio = rr.get("ratio"), rp.get("ratio")
        rg, mg = _gap(rep_ratio, up), _gap(meas_ratio, up)
        rows.append(dict(
            pr=pr, field=d.get("field"), hardware=d.get("hw_slug") or os.path.basename(p).split(".")[1],
            headline=s.get("headline"), unit=h.get("unit"),
            reported_baseline=rr.get("baseline"), reported_pr_solution=rr.get("pr_solution"), reported_marker=rr.get("marker"),
            reported_ratio=rep_ratio, reported_gap=rg,
            reproduced_baseline=rp.get("baseline"), reproduced_pr_solution=rp.get("pr_solution"),
            reproduced_ratio=meas_ratio, reproduced_ratio_sd=rp.get("ratio_sd"), reproduced_gap=mg,
            gap_ratio=(mg / rg) if (rg and np.isfinite(rg) and np.isfinite(mg)) else np.nan,
            verdict=s.get("verdict"), reliability=(h.get("reliability") or {}).get("flag"),
            tests=f"{(s.get('tests') or {}).get('base')}/{(s.get('tests') or {}).get('gold')}", title=d.get("title")))
    return pd.DataFrame(rows, columns=PR_COLS).sort_values("pr").reset_index(drop=True)


def per_field(df):
    if df.empty:
        return pd.DataFrame(columns=["n", "gap_ratio_med", "gap_ratio_min", "gap_ratio_max", "reported_gap_med", "reproduced_gap_med"])
    g = df.groupby("field")
    return pd.DataFrame({"n": g.size(), "gap_ratio_med": g.gap_ratio.median(), "gap_ratio_min": g.gap_ratio.min(),
                         "gap_ratio_max": g.gap_ratio.max(), "reported_gap_med": g.reported_gap.median(),
                         "reproduced_gap_med": g.reproduced_gap.median()}).sort_index()


def status_rows():
    from pipeline import next_prs
    out = []
    for pr in layout.workload_prs():
        st, why = next_prs.state(pr)
        t = layout.load_task(pr)
        out.append({"pr": pr, "field": t.get("field"), "hardware": layout.hw_slug(), "status": st, "reason": " ".join(why), "title": t.get("title")})
    return out


def plot(df, pf, out_png):
    import matplotlib.pyplot as plt   # backend: Agg from main(), inline inside the notebook
    x = np.arange(len(df)); lab = [f"#{p}" for p in df.pr]
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [3, 2]})
    w = .38
    # 1. gap ratio per PR (log y, 1 = exactly the author's gap; bars grow from the 1x reference)
    a = ax[0, 0]; m = df.gap_ratio.notna() & (df.gap_ratio > 0)
    a.bar(x[m], df.gap_ratio[m] - 1, bottom=1, color=BLUE, width=.6)
    a.set_yscale("log", base=2); a.axhline(1, color="#c9d6ec", lw=1.4); a.set_xticks(x); a.set_xticklabels(lab, rotation=55, ha="right", fontsize=9)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×")); a.set_title("gap ratio: reproduced gap ÷ reported gap", loc="left", fontweight="bold")
    flag = df.reliability.eq("below_host_floor") & m
    for xi in x[flag]:
        a.bar(xi, df.gap_ratio[xi] - 1, bottom=1, color="none", edgecolor="#e5533f", linestyle="--", width=.6)
    # 2. gaps per PR: reported vs reproduced, relative to baseline
    a = ax[0, 1]
    a.bar(x - w / 2, df.reported_gap.fillna(0) * 100, w, color=ORANGE, label="reported gap")
    a.bar(x + w / 2, df.reproduced_gap.fillna(0) * 100, w, color=BLUE, label="reproduced gap")
    a.axhline(0, color="#c9d6ec", lw=1.4); a.set_xticks(x); a.set_xticklabels(lab, rotation=55, ha="right", fontsize=9)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%")); a.set_title("gaps: (baseline − PR's solution) ÷ baseline", loc="left", fontweight="bold"); a.legend(frameon=False, fontsize=8)
    # 3. gap ratio per field (median, whisker = min–max)
    fx = np.arange(len(pf))
    a = ax[1, 0]; mm = pf.gap_ratio_med.notna() & (pf.gap_ratio_med > 0)
    if mm.any():
        a.bar(fx[mm.values], pf.gap_ratio_med[mm] - 1, bottom=1, color=BLUE, width=.5)
        a.errorbar(fx[mm.values], pf.gap_ratio_med[mm], yerr=[(pf.gap_ratio_med - pf.gap_ratio_min)[mm], (pf.gap_ratio_max - pf.gap_ratio_med)[mm]], fmt="none", ecolor="#172a5c", capsize=3)
        a.set_yscale("log", base=2)
    a.axhline(1, color="#c9d6ec", lw=1.4); a.set_xticks(fx); a.set_xticklabels([f"{f} (n={n})" for f, n in zip(pf.index, pf.n)], rotation=35, ha="right", fontsize=9)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}×")); a.set_title("gap ratio per field (median, whisker = min–max)", loc="left", fontweight="bold")
    # 4. gaps per field (median)
    a = ax[1, 1]
    a.bar(fx - w / 2, pf.reported_gap_med.fillna(0) * 100, w, color=ORANGE, label="reported gap (median)")
    a.bar(fx + w / 2, pf.reproduced_gap_med.fillna(0) * 100, w, color=BLUE, label="reproduced gap (median)")
    a.axhline(0, color="#c9d6ec", lw=1.4); a.set_xticks(fx); a.set_xticklabels(list(pf.index), rotation=35, ha="right", fontsize=9)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%")); a.set_title("gaps per field (median)", loc="left", fontweight="bold"); a.legend(frameon=False, fontsize=8)
    for aa in ax.flat:
        aa.grid(axis="y", color="#eef2f9"); aa.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=110, metadata={"Software": None})   # no mtime/software stamp: unchanged data = unchanged file
    return fig


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    a = sys.argv[1:]
    hw = a[a.index("--hw") + 1] if "--hw" in a else None
    out = a[a.index("--out") + 1] if "--out" in a else OUT
    os.makedirs(out, exist_ok=True)
    df = load_prs(hw)
    pf = per_field(df)
    df.to_csv(os.path.join(out, "prs.csv"), index=False, float_format="%.6g")
    pf.reset_index().rename(columns={"index": "field"}).to_csv(os.path.join(out, "fields.csv"), index=False, float_format="%.6g")
    st = status_rows()
    with open(os.path.join(out, "status.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["pr", "field", "hardware", "status", "reason", "title"])
        wr.writeheader(); wr.writerows(st)
    plot(df, pf, os.path.join(out, "gaps.png"))
    n_ok = int((df.verdict == "reproduced").sum()) if not df.empty else 0
    print(f"{len(df)} PRs reproduced ({hw or layout.hw_slug()}) · {n_ok} match the reported ratio (≥0.8 of it) · "
          f"{sum(1 for r in st if r['status'] == 'measured')}/{len(st)} workload PRs measured -> {layout.rel(out)}/{{prs,fields,status}}.csv + gaps.png")


if __name__ == "__main__":
    main()
