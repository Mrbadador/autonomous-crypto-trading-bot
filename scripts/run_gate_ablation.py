import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from datetime import datetime

from backtests.backtest_ablation_harness import (
    precompute_4h_indicators_vectorized,
    precompute_1d_trend_vectorized,
    run_ablation_simulation,
)

EXPERIMENTS = [
    ("Full Stack (All Gates ON)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": True}),
    ("Ablate Gate A (1D Trend Support OFF)", {"gate_1d_trend": False, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": True}),
    ("Ablate Gate B (Wick Exhaustion OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": False, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": True}),
    ("Ablate Gate C (Momentum Divergence OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": False, "gate_td9": True, "gate_volume_profile": True}),
    ("Ablate Gate D (TD-9 Exhaustion OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": False, "gate_volume_profile": True}),
    ("Ablate Gate E (Volume Profile OFF)", {"gate_1d_trend": True, "gate_wick_exhaustion": True, "gate_divergence": True, "gate_td9": True, "gate_volume_profile": False}),
    ("Naked Baseline (All Gates OFF)", {"gate_1d_trend": False, "gate_wick_exhaustion": False, "gate_divergence": False, "gate_td9": False, "gate_volume_profile": False}),
]

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

FULL_STACK_NAME = "Full Stack (All Gates ON)"

# (heading, config flag ablated, ablation arm name). Each gate section is wired to its OWN
# ablation arm: the writer never borrows a neighbouring gate's comparison.
GATE_SECTIONS = [
    ("1. Gate A — 1D EMA Trend Alignment (`gate_1d_trend`)",
     "gate_1d_trend",
     "Ablate Gate A (1D Trend Support OFF)"),
    ("2. Gate C — Momentum Divergence (`gate_divergence`)",
     "gate_divergence",
     "Ablate Gate C (Momentum Divergence OFF)"),
    ("3. Gate B — Wick Exhaustion (`gate_wick_exhaustion`)",
     "gate_wick_exhaustion",
     "Ablate Gate B (Wick Exhaustion OFF)"),
    ("4. Gate D — TD-9 Count Exhaustion (`gate_td9`)",
     "gate_td9",
     "Ablate Gate D (TD-9 Exhaustion OFF)"),
    ("5. Gate E — Volume Profile & Value Area Clearance (`gate_volume_profile`)",
     "gate_volume_profile",
     "Ablate Gate E (Volume Profile OFF)"),
]

MIN_BOOTSTRAP_TRADES = 5


def _bootstrap_delta_ev(full_pnls, ablated_pnls, n_boot=1500, seed=42, alpha=10.0):
    """90% bootstrap CI for (mean gate-ON PnL - mean gate-OFF PnL), or (None, None).

    Each arm's closed-trade PnL series is resampled iid 1500 times with a fixed seed, so the
    same inputs always yield the same interval. The two arms are resampled INDEPENDENTLY even
    though they replay the same market path, which makes the interval wider than a paired or
    block bootstrap would give: the design errs toward INCONCLUSIVE rather than toward a false
    KEEP. Returns (None, None) when either arm is too small to difference.
    """
    rng = np.random.default_rng(seed)
    if len(full_pnls) < MIN_BOOTSTRAP_TRADES or len(ablated_pnls) < MIN_BOOTSTRAP_TRADES:
        return None, None
    deltas = []
    for _ in range(n_boot):
        f = rng.choice(full_pnls, size=len(full_pnls), replace=True)
        a = rng.choice(ablated_pnls, size=len(ablated_pnls), replace=True)
        deltas.append(float(np.mean(f) - np.mean(a)))
    return float(np.percentile(deltas, alpha / 2)), float(np.percentile(deltas, 100 - alpha / 2))


def verdict_for_gate(full, ablated):
    """Return (verdict, evidence) for one gate, derived from the bootstrap CI of ΔEV.

    ``full`` and ``ablated`` are arm dicts from run_ablation_simulation (``trades`` is a list of
    dicts carrying ``net_pnl``); either may be None/absent when that arm did not run.
    """
    if not full or not ablated:
        if not full and not ablated:
            missing = "gate-ON reference and gate-OFF ablation"
        elif not full:
            missing = "gate-ON reference"
        else:
            missing = "gate-OFF ablation"
        return "INCONCLUSIVE", (
            f"ΔEV is not estimable: the {missing} arm did not run, so this section has no measured "
            "comparison and reports nothing about the gate."
        )

    full_pnls = [t["net_pnl"] for t in full.get("trades", [])]
    ablated_pnls = [t["net_pnl"] for t in ablated.get("trades", [])]
    lo, hi = _bootstrap_delta_ev(full_pnls, ablated_pnls)

    if lo is None or hi is None:
        # No interval exists, so none is printed: a fabricated [+0.000%, +0.000%] would read as a
        # precise measurement of "no effect". The real arm sizes are named instead.
        return "INCONCLUSIVE", (
            f"ΔEV is not estimable: the gate-ON arm closed {len(full_pnls)} trades and the gate-OFF arm "
            f"closed {len(ablated_pnls)}, below the {MIN_BOOTSTRAP_TRADES}-per-arm minimum a bootstrap "
            "difference in means needs. No ΔEV and no 90% interval are reported for this pair."
        )

    delta = float(np.mean(full_pnls) - np.mean(ablated_pnls))
    evidence = (
        f"ΔEV (gate ON minus gate OFF) = {delta * 100:+.3f}% per trade, "
        f"90% CI [{lo * 100:+.3f}%, {hi * 100:+.3f}%] "
        f"over N={len(full_pnls)}/{len(ablated_pnls)} trades"
    )
    if lo > 0:
        return "KEEP", evidence + "; removing this gate lowers expected value and the 90% CI excludes zero."
    if hi < 0:
        return "RETHINK", evidence + "; removing this gate raises expected value and the 90% CI excludes zero."
    return "INCONCLUSIVE", (
        evidence + "; the 90% CI for ΔEV straddles zero, so this sample cannot measure the gate's contribution."
    )


def evaluate_all():
    all_results = {}
    for sym in SYMBOLS:
        csv_4h = f"data/historical/{sym}_4h.csv"
        csv_1d = f"data/historical/{sym}_1d.csv"
        if not (os.path.exists(csv_4h) and os.path.exists(csv_1d)):
            print(f"[!] Missing data for {sym}, skipping")
            continue
        
        df_4h = pd.read_csv(csv_4h)
        df_1d = pd.read_csv(csv_1d)
        for col in ["open", "high", "low", "close", "volume"]:
            df_4h[col] = df_4h[col].astype(float)
            df_1d[col] = df_1d[col].astype(float)
            
        df_4h_vec = precompute_4h_indicators_vectorized(df_4h)
        df_1d_vec = precompute_1d_trend_vectorized(df_1d)
        
        sym_results = []
        for name, cfg in EXPERIMENTS:
            res = run_ablation_simulation(
                df_4h, df_1d, cfg, df_4h_vec=df_4h_vec, df_1d_vec=df_1d_vec
            )
            sym_results.append({
                "name": name,
                "n_trades": res["n_trades"],
                "win_rate": res["win_rate"],
                "total_return": res["total_return"],
                "profit_factor": res["profit_factor"],
                "sharpe": res["sharpe"],
                "ev_mean": res["ev_mean"],
                "ev_ci_90": res["ev_ci_90"],
                # Kept for the ΔEV bootstrap: the verdicts need each arm's per-trade PnL series and
                # the count of entries the gate actually vetoed.
                "trades": res["trades"],
                "gate_fires": res["gate_fires"],
            })
        all_results[sym] = sym_results

    # Write report
    report_path = f"audits/gate_ablation_report_{datetime.now().strftime('%Y-%m-%d')}.md"
    os.makedirs("audits", exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Quantitative Gate Ablation Report & Empirical Verdicts\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("**Methodology:** 2-Year Authentic Historical OHLCV Replay (Bitget Futures). Full Causal indicators with Bootstrap 90% Confidence Interval for Expected Value (EV) per trade.\n\n")
        f.write("---\n\n")
        
        for sym, rows in all_results.items():
            f.write(f"## Asset: {sym}\n\n")
            f.write("| Gate Configuration | Trades (N) | Win Rate (%) | Cumulative Return (%) | Profit Factor | Sharpe | EV / Trade (90% CI) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in rows:
                ci = f"{r['ev_mean']:+.2f}% [{r['ev_ci_90'][0]:+.2f}%, {r['ev_ci_90'][1]:+.2f}%]"
                f.write(f"| {r['name']} | {r['n_trades']} | {r['win_rate']:.1f}% | {r['total_return']:+.2f}% | {r['profit_factor']:.2f} | {r['sharpe']:.2f} | {ci} |\n")
            f.write("\n")
            
        # One row lookup per asset: each gate section below reads its OWN ablation arm.
        rows_by_symbol = {sym: {r["name"]: r for r in all_results.get(sym, [])} for sym in SYMBOLS}

        f.write("## Statistical Analysis & Empirical Verdicts\n\n")
        f.write(
            "Every verdict in this section is derived from the harness's own per-trade PnL series: a "
            "1500-resample iid bootstrap (seed 42) of the difference in mean net PnL per trade between the "
            "full gate stack and that one gate removed, reported as a 90% confidence interval of ΔEV. "
            "KEEP requires the interval to lie entirely above zero, RETHINK entirely below it, and anything "
            "else — including an interval that straddles zero, an arm too small to difference, or an "
            "ablation arm that never ran — is INCONCLUSIVE. No verdict is written independently of the "
            "interval it comes from.\n\n"
        )

        for heading, param_name, ablated_name in GATE_SECTIONS:
            f.write(f"### {heading}\n")
            for sym in SYMBOLS:
                rows = rows_by_symbol.get(sym, {})
                full_row = rows.get(FULL_STACK_NAME)
                ablated_row = rows.get(ablated_name)
                verdict, evidence = verdict_for_gate(full_row, ablated_row)
                f.write(f"- **{sym} — Verdict:** **{verdict}**\n")
                f.write(f"- **{sym} — Empirical Evidence:** {evidence}\n")
                fires = (full_row or {}).get("gate_fires", {}).get(param_name)
                if fires is None:
                    f.write(
                        f"- **{sym} — Veto count in the full stack:** not recorded for `{param_name}`.\n"
                    )
                elif fires == 0:
                    f.write(
                        f"- **{sym} — Veto count in the full stack:** 0 candidate entries were rejected by "
                        f"`{param_name}`, so turning it off changes no entry decision; the ΔEV above is "
                        f"structural, not a measured estimate of the gate's value.\n"
                    )
                else:
                    noun = "entry" if fires == 1 else "entries"
                    f.write(
                        f"- **{sym} — Veto count in the full stack:** {fires} candidate {noun} rejected by "
                        f"`{param_name}` (gate-ON N={full_row.get('n_trades')}, "
                        f"gate-OFF N={(ablated_row or {}).get('n_trades')}).\n"
                    )
            f.write("\n")

        f.write("---\n\n")
        f.write("### Methodology caveats\n\n")
        f.write("- Confidence intervals are an iid bootstrap over closed trades and do not correct for serial dependence between consecutive entries.\n")
        f.write("- The ΔEV interval resamples the gate-ON and gate-OFF arms independently even though both arms replay the same market path, so it is wider than a paired or block bootstrap would give. The method is deliberately conservative — it errs toward INCONCLUSIVE rather than toward a false KEEP — and the interval should be read as approximate, not exact.\n\n")
        f.write("*Autonomously generated by Quantitative Ablation Harness.*")

    print(f"[OK] Report written to {report_path}")

if __name__ == "__main__":
    evaluate_all()
