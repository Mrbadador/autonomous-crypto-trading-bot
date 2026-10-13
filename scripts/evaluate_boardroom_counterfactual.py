"""
scripts/evaluate_boardroom_counterfactual.py

Boardroom Counterfactual Scoring & Conviction Calibration Evaluator (FIX 1 — honest edition).

Scores multi-agent LLM deliberations (`boardroom_debates` from trading_ledger.db) on
post-cutoff decisions (>= 2026-10-01).

MODEL PROVENANCE: the serving model per debate was not persisted before FIX 2, so
decisions cannot be attributed to any training cutoff. Scoring is presented as
pool-level statistics only; see report header.

Method:
1. Approved-executed population: debates (execute=1) matched to realized `swing_trades` PnL.
2. Vetoed population (shadow): for each vetoed debate, reconstruct the would-be swing setup
   (entry = completed 4h signal-candle close at debate time; SL = max(3.0xATR14(4h), 1.5%);
   TP = 4.0x risk — swing_trader.py:1032-1046 geometry), then replay it through the
   EXISTING fee+funding TP/SL state machine in backtests/backtest_grid_replay.py
   (seeded_positions mode; taker entry, maker TP, taker SL, real Bitget 8h funding).
3. Boardroom value = approved realized EV vs (approved + vetoed-shadow) combined EV,
   as a difference of two measured populations with a joint bootstrap 90% CI.
   This is NOT "capital defended" — that fabricated metric was withdrawn 2026-10-14.

Market data: Bitget public REST (live venue), cached under data/counterfactual/ by
scripts/fetch_counterfactual_data.py. No order endpoints are touched.

EVERY FIGURE IS A SHADOW/BACKTEST ESTIMATE. Fill model 'touch' is an UPPER BOUND on
passive-fill realism (queue position unmodeled; queue-model calibration blocked on
TRIAL-003 books15 accumulation — see audits/prereg_registry.md).
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, UTC, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtests.backtest_grid_replay import run_grid_replay

POST_CUTOFF_DATE = "2026-10-01T00:00:00"

# Swing strategy geometry (config.py values read 2026-10-13; swing_trader.py:1032-1046):
SWING_ATR_SL_MULT = 3.0        # config.SWING_ATR_SL_MULT
SWING_MIN_SL_PCT = 0.015       # config.SWING_MIN_SL_PCT
SWING_RUNNER_TP_RR = 4.0       # config.SWING_RUNNER_TP_MULT (TP = 4.0 x risk)
ATR_WINDOW = 14

# Standardized shadow economics (labeled assumptions, see report §Assumptions):
SHADOW_ALLOCATION_USD = 85.0   # config SWING_TRADE_ALLOCATION_USDT default (swing_trader.py:59)
SHADOW_LEVERAGE = 3.0          # config.SWING_STANDARD_LEVERAGE
SHADOW_NOTIONAL_USD = SHADOW_ALLOCATION_USD * SHADOW_LEVERAGE  # 255.0

BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42


def bootstrap_ci(values, ci_level: float = 0.90) -> Optional[dict]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return {"mean": mean, "n": len(arr), "lower": None, "upper": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(BOOTSTRAP_DRAWS)])
    alpha = (1.0 - ci_level) / 2.0
    return {
        "mean": mean,
        "n": len(arr),
        "lower": float(np.percentile(boot, alpha * 100)),
        "upper": float(np.percentile(boot, (1.0 - alpha) * 100)),
    }


def bootstrap_diff_ci(a, b, ci_level: float = 0.90) -> Optional[dict]:
    """CI of mean(a) - mean(b) via independent joint resampling of both populations."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(BOOTSTRAP_DRAWS)
    for k in range(BOOTSTRAP_DRAWS):
        diffs[k] = np.mean(rng.choice(a, len(a), True)) - np.mean(rng.choice(b, len(b), True))
    alpha = (1.0 - ci_level) / 2.0
    return {
        "diff": float(np.mean(a) - np.mean(b)),
        "n_a": len(a),
        "n_b": len(b),
        "lower": float(np.percentile(diffs, alpha * 100)),
        "upper": float(np.percentile(diffs, (1.0 - alpha) * 100)),
    }


def read_debates(ledger_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{ledger_db}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(boardroom_debates)")]
        prov = ", served_model_id" if "served_model_id" in cols else ", NULL AS served_model_id"
        df = pd.read_sql_query(
            f"SELECT id, symbol, side, execute, conviction_score, final_rationale, timestamp{prov} "
            f"FROM boardroom_debates WHERE timestamp >= '{POST_CUTOFF_DATE}' ORDER BY timestamp ASC",
            conn,
        )
    finally:
        conn.close()
    return df


def read_trades(data_db: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{data_db}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT id, timestamp, symbol, side, entry_price, size, pnl, exit_reason, status "
            f"FROM swing_trades WHERE timestamp >= '{POST_CUTOFF_DATE}' ORDER BY timestamp ASC",
            conn,
        )
    finally:
        conn.close()
    return df


def _cache_key(base: str) -> str:
    for suf in ("_5m", "_4h", "_funding"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base[:-4] if base.endswith("USDT") else base


def load_cache(cache_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    data: Dict[str, Dict[str, pd.DataFrame]] = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".csv"):
            continue
        base = fname[:-4]
        df = pd.read_csv(os.path.join(cache_dir, fname))
        df["timestamp"] = df["timestamp"].astype("int64")
        kind = "5m" if base.endswith("_5m") else "4h" if base.endswith("_4h") else "funding" if base.endswith("_funding") else None
        if kind is None:
            continue
        data.setdefault(_cache_key(base), {})[kind] = df
    return data


def wilder_atr(df_4h: pd.DataFrame, idx: int, window: int = ATR_WINDOW) -> float:
    """Wilder RMA ATR(14) over 4h bars [0..idx] inclusive. NaN when history too short."""
    if idx + 1 < window + 1:
        return float("nan")
    h, l, c = df_4h["high"].values, df_4h["low"].values, df_4h["close"].values
    hh, ll, pc = h[1: idx + 1], l[1: idx + 1], c[: idx + 1][:-1]
    tr = np.maximum(hh - ll, np.maximum(np.abs(hh - pc), np.abs(ll - pc)))
    atr = float(np.mean(tr[:window]))
    for x in tr[window:]:
        atr = (atr * (window - 1) + x) / window
    return atr


def derive_setups(vetoed_df: pd.DataFrame, cache: Dict[str, Dict[str, pd.DataFrame]]) -> tuple:
    """Reconstruct would-be swing setups for vetoed debates (no-lookahead: only completed 4h bars)."""
    setups = []
    skipped = {"no_cache": 0, "no_completed_4h_bar": 0, "atr_unavailable": 0,
               "entry_bar_missing": 0, "entry_mismatch_gt_0.2pct": 0, "geometry_invalid": 0}
    for row in vetoed_df.itertuples():
        unit = _cache_key(str(row.symbol).upper().replace("/", ""))
        side_raw = str(row.side).upper()
        side = "LONG" if side_raw in ("LONG", "BUY", "L") else "SHORT"
        ts = pd.to_datetime(row.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_ms = int(ts.timestamp() * 1000)
        c = cache.get(unit)
        if not c or "5m" not in c or "4h" not in c:
            skipped["no_cache"] += 1
            continue
        df4 = c["4h"]
        closed = df4[(df4["timestamp"] + 4 * 3600_000) <= ts_ms]
        if closed.empty:
            skipped["no_completed_4h_bar"] += 1
            continue
        sig_idx = int(closed.index[-1])
        entry_price_4h = float(df4.loc[sig_idx, "close"])
        atr = wilder_atr(df4, sig_idx)
        if np.isnan(atr) or atr <= 0:
            skipped["atr_unavailable"] += 1
            continue
        sl_dist = max(SWING_ATR_SL_MULT * atr, entry_price_4h * SWING_MIN_SL_PCT)
        if side == "LONG":
            sl, tp = entry_price_4h - sl_dist, entry_price_4h + SWING_RUNNER_TP_RR * sl_dist
        else:
            sl, tp = entry_price_4h + sl_dist, entry_price_4h - SWING_RUNNER_TP_RR * sl_dist
        if sl <= 0 or tp <= 0:
            skipped["geometry_invalid"] += 1
            continue
        if side == "LONG" and not (sl < entry_price_4h < tp):
            skipped["geometry_invalid"] += 1
            continue
        if side == "SHORT" and not (tp < entry_price_4h < sl):
            skipped["geometry_invalid"] += 1
            continue
        df5 = c["5m"]
        entry_bar_ts = int(df4.loc[sig_idx, "timestamp"]) + 4 * 3600_000 - 5 * 60_000
        match = df5[df5["timestamp"] == entry_bar_ts]
        if match.empty:
            skipped["entry_bar_missing"] += 1
            continue
        entry_price = float(match["close"].iloc[0])
        if abs(entry_price - entry_price_4h) / max(entry_price_4h, 1e-9) > 0.002:
            skipped["entry_mismatch_gt_0.2pct"] += 1
            continue
        setups.append({
            "debate_id": int(row.id),
            "symbol": f"{unit}USDT",
            "side": side,
            "entry_price": entry_price,
            "take_profit": tp,
            "stop_loss": sl,
            "entry_ts": entry_bar_ts,
            "atr_4h": atr,
            "label": f"debate#{row.id} {unit} {side} {row.timestamp}",
        })
    return setups, skipped


def replay_setup(setup: dict, cache: dict, fill_model: str = "touch") -> dict:
    """Replay one reconstructed setup through the existing fee+funding TP/SL state machine."""
    unit = _cache_key(setup["symbol"])
    c = cache[unit]
    df = c["5m"]
    pos = np.where(df["timestamp"].values == setup["entry_ts"])[0]
    if len(pos) == 0:
        return {**setup, "filled_and_closed": False, "exit_reason": "ENTRY_BAR_MISSING",
                "net_pnl_usd": float("nan"), "unresolved": True}
    i0 = int(pos[0])
    window = df.iloc[max(0, i0 - 30):].reset_index(drop=True)
    trades = run_grid_replay(
        window,
        symbol=setup["symbol"],
        funding_df=c.get("funding"),
        btc_15m_df=None,
        fill_model=fill_model,
        enable_45m_tightening=False,
        trade_allocation_usd=SHADOW_ALLOCATION_USD,
        leverage=SHADOW_LEVERAGE,
        scan_signals=False,
        seeded_positions=[{k: setup[k] for k in ("side", "entry_price", "take_profit", "stop_loss", "entry_ts", "symbol", "label")}],
    )
    res = {**setup}
    if trades:
        t = trades[0]
        res.update({
            "filled_and_closed": True,
            "exit_reason": t["exit_reason"],
            "exit_price": t.get("exit_price"),
            "exit_ts": t.get("exit_ts"),
            "holding_mins": t.get("holding_mins"),
            "gross_pnl": t.get("gross_pnl"),
            "net_pnl_usd": t.get("net_pnl"),
            "funding_usd": t.get("funding_usd", 0.0),
            "fees_usd": float(t.get("entry_fee_usd", 0.0)) + float(t.get("exit_fee_usd", 0.0)),
            "unresolved": bool(t.get("unresolved", False)),
        })
    else:
        res.update({"filled_and_closed": False, "exit_reason": "SEED_NOT_PLACED",
                    "net_pnl_usd": float("nan"), "unresolved": True})
    return res


def match_approved(debates_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Match approved debates to realized swing_trades rows (12h forward window, 1:1 consumption)."""
    approved = debates_df[debates_df["execute"] == 1].copy()
    approved["ts"] = pd.to_datetime(approved["timestamp"], format="mixed", utc=True)
    tr = trades_df.copy()
    tr["ts"] = pd.to_datetime(tr["timestamp"], format="mixed", utc=True)
    tr["sym_key"] = tr["symbol"].str.upper().str.replace("/", "", regex=False)
    matched, used = [], set()
    for row in approved.itertuples():
        key = str(row.symbol).upper().replace("/", "")
        cand = tr[
            (tr["sym_key"] == key) &
            (tr["ts"] >= row.ts - pd.Timedelta(minutes=90)) &
            (tr["ts"] <= row.ts + pd.Timedelta(hours=12)) &
            (~tr["id"].isin(used))
        ]
        if cand.empty:
            continue
        pick = cand.iloc[0]
        used.add(int(pick["id"]))
        matched.append({
            "debate_id": int(row.id),
            "symbol": row.symbol,
            "side": row.side,
            "conviction": float(row.conviction_score),
            "trade_id": int(pick["id"]),
            "realized_pnl_usd": float(pick["pnl"]) if pd.notna(pick["pnl"]) else 0.0,
            "exit_reason": pick["exit_reason"],
        })
    return pd.DataFrame(matched)


def summarize(pnls, n_total: int) -> dict:
    arr = np.asarray(pnls, dtype=float)
    clean = arr[~np.isnan(arr)]
    return {
        "n_total": n_total,
        "n_measured": int(len(clean)),
        "n_unresolved": int(len(arr) - len(clean)),
        "ev": (float(np.mean(clean)) if len(clean) else None),
        "ci": bootstrap_ci(clean),
        "win_rate": (float(np.sum(clean > 0) / len(clean) * 100.0) if len(clean) else None),
        "total_pnl": (float(np.sum(clean)) if len(clean) else None),
    }


def evaluate(ledger_db: str, data_db: str, cache_dir: str) -> Dict[str, Any]:
    debates = read_debates(ledger_db)
    trades = read_trades(data_db)
    cache = load_cache(cache_dir)

    approved_df = debates[debates["execute"] == 1]
    vetoed_df = debates[debates["execute"] == 0]

    setups, skipped = derive_setups(vetoed_df, cache)
    # Deduplicate: the boardroom re-deliberates the same live signal every few minutes.
    # Statistics must treat one setup per (symbol, entry candle, side) or the bootstrap CI is falsely tight.
    unique: Dict[tuple, dict] = {}
    for s in setups:
        key = (s["symbol"], s["entry_ts"], s["side"])
        if key not in unique:
            unique[key] = s
    dup_count = len(setups) - len(unique)
    setups_u = list(unique.values())
    shadow_touch = [replay_setup(s, cache, fill_model="touch") for s in setups_u]
    shadow_1tick = [replay_setup(s, cache, fill_model="one_tick") for s in setups_u]
    shadow_pnl = np.array([r["net_pnl_usd"] for r in shadow_touch], dtype=float)
    shadow_pnl_1t = np.array([r["net_pnl_usd"] for r in shadow_1tick], dtype=float)

    exit_counts: Dict[str, int] = {}
    for r in shadow_touch:
        if r.get("filled_and_closed"):
            exit_counts[r["exit_reason"]] = exit_counts.get(r["exit_reason"], 0) + 1

    vetoed_pop = summarize(shadow_pnl, len(vetoed_df))
    vetoed_pop["exit_reasons"] = exit_counts
    vetoed_pop["skipped"] = skipped
    vetoed_pop["reconstructed"] = len(setups)
    vetoed_pop["unique_setups"] = len(setups_u)
    vetoed_pop["duplicates_dropped"] = dup_count
    vetoed_pop_1tick = summarize(shadow_pnl_1t, len(setups_u))

    matched = match_approved(debates, trades)
    approved_pnl = matched["realized_pnl_usd"].values.astype(float) if len(matched) else np.array([])
    approved_pop = summarize(approved_pnl, len(approved_df))

    combined = np.concatenate([approved_pnl[~np.isnan(approved_pnl)], shadow_pnl[~np.isnan(shadow_pnl)]])
    combined_pop = summarize(combined, len(approved_df) + len(vetoed_df))
    bv = bootstrap_diff_ci(approved_pnl, combined)

    calib = []
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]:
        sub_debates = debates[(debates["conviction_score"] >= lo) & (debates["conviction_score"] < hi)]
        sub = matched[matched["conviction"] >= lo] if len(matched) else matched
        if len(sub):
            sub = sub[sub["conviction"] < hi]
        arr = sub["realized_pnl_usd"].values.astype(float) if len(sub) else np.array([])
        calib.append({"bin": f"{lo:.2f}–{hi:.2f}", "deliberations": len(sub_debates),
                      "matched": int(len(arr)), "pop": summarize(arr, len(arr))})

    n_prov = int(debates["served_model_id"].notna().sum()) if "served_model_id" in debates.columns else 0

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "post_cutoff": POST_CUTOFF_DATE,
        "n_debates": int(len(debates)),
        "n_approved": int(len(approved_df)),
        "n_vetoed": int(len(vetoed_df)),
        "n_provenance_rows": n_prov,
        "vetoed_shadow": vetoed_pop,
        "vetoed_shadow_1tick": vetoed_pop_1tick,
        "approved_realized": approved_pop,
        "combined_if_no_boardroom": combined_pop,
        "boardroom_value": bv,
        "calibration": calib,
        "matched_trades": matched.to_dict("records") if len(matched) else [],
        "shadow_rows": shadow_touch,
        "skipped": skipped,
    }


def money(v):
    return "NOT MEASURABLE" if v is None else f"${v:+.3f}"


def ci_str(pop):
    ci = pop.get("ci") if isinstance(pop, dict) else None
    if not ci or ci.get("lower") is None:
        return "NOT MEASURABLE (N < 2)" if ci else "NOT MEASURABLE (N = 0)"
    return f"[{ci['lower']:+.3f}, {ci['upper']:+.3f}]"


def generate_report(res: Dict[str, Any], out_path: str, dump_csv: Optional[str] = None):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    v, v1, a, c = res["vetoed_shadow"], res["vetoed_shadow_1tick"], res["approved_realized"], res["combined_if_no_boardroom"]
    bv = res["boardroom_value"]
    lines = []
    w = lines.append
    w("# LLM Boardroom Counterfactual Audit — Shadow Replay (HONEST EDITION)")
    w("")
    w(f"**Generated:** {res['generated_at']}  ")
    w(f"**Data horizon:** ledger rows with timestamp >= {res['post_cutoff']} (decision log, NOT training-cutoff verified)  ")
    if res["n_provenance_rows"] < 50:
        w(f"**MODEL PROVENANCE: unpinned pool; post-cutoff scoring paused** — {res['n_provenance_rows']}/50 debates carry `served_model_id`; decisions cannot yet be attributed to a model training cutoff.  ")
    else:
        w(f"**MODEL PROVENANCE:** {res['n_provenance_rows']} debates carry `served_model_id`.  ")
    w("")
    w("> [!WARNING]")
    w("> **All figures are SHADOW/BACKTEST ESTIMATES. The `touch` fill model is an UPPER BOUND** (best queue position")
    w("> assumed for passive legs; queue-model calibration is blocked on TRIAL-003 books15 depth accumulation).")
    w("> The prior report's **\"Estimated Capital Defended $+301.76\"** and **\"Boardroom-OFF ablation −$176.02\"** were")
    w("> hardcoded arithmetic (`368 × $0.82`; `approved_pnl − 368 × $0.45`) with no simulation behind them.")
    w("> **WITHDRAWN 2026-10-14: those figures were fabricated constants, not measurements.**")
    w("")
    w("## 1. Method")
    w("")
    w("- Vetoed debates → reconstructed swing setups (entry = completed 4h signal-candle close;")
    w("  SL = max(3.0×ATR14(4h), 1.5%); TP = 4.0×risk — swing_trader.py:1032-1046 geometry, config defaults read 2026-10-13).")
    w("- Replayed through `backtests/backtest_grid_replay.py::run_grid_replay` `seeded_positions` mode: the same")
    w("  fee-inclusive TP/SL/funding state machine as the verified grid harness (taker 0.06% entry, maker 0.02% TP,")
    w("  taker 0.06% SL, real Bitget 8h funding history).")
    w("- Shadow economics standardized at $85 margin x 3x leverage = $255 notional (labeled assumption; live executed median notional measured $181.34 from swing_trades).")
    w("- Market data: Bitget public REST cached 2026-10-13 by `scripts/fetch_counterfactual_data.py`.")
    w("- Approved population: realized `swing_trades.pnl` (ledger USD, mixed notionals).")
    w("")
    w("## 2. Two measured populations")
    w("")
    w("| Population | N debates | N replayed/matched (unique) | N unresolved | EV/trade (USD) | Bootstrap 90% CI | Win rate | Total PnL |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    w(f"| (i) Vetoed → shadow (touch, UPPER BOUND) | {res['n_vetoed']} | {v['n_measured']} | {v['n_unresolved']} | {money(v['ev'])} | {ci_str(v)} | {('%.1f%%' % v['win_rate']) if v['win_rate'] is not None else 'NOT MEASURABLE'} | {money(v['total_pnl'])} |")
    w(f"| (i-b) same, 1-tick conservative fills | {res['n_vetoed']} | {v1['n_measured']} | {v1['n_unresolved']} | {money(v1['ev'])} | {ci_str(v1)} | {('%.1f%%' % v1['win_rate']) if v1['win_rate'] is not None else 'NOT MEASURABLE'} | {money(v1['total_pnl'])} |")
    w(f"| (ii) Approved → executed actuals | {res['n_approved']} | {a['n_measured']} | {a['n_unresolved']} | {money(a['ev'])} | {ci_str(a)} | {('%.1f%%' % a['win_rate']) if a['win_rate'] is not None else 'NOT MEASURABLE'} | {money(a['total_pnl'])} |")
    w(f"| (iii) Combined = approved + vetoed-shadow (Boardroom-OFF proxy) | {res['n_approved'] + res['n_vetoed']} | {c['n_measured']} | {c['n_unresolved']} | {money(c['ev'])} | {ci_str(c)} | {('%.1f%%' % c['win_rate']) if c['win_rate'] is not None else 'NOT MEASURABLE'} | {money(c['total_pnl'])} |")
    w("")
    w("*`touch` and `one_tick` coincide whenever every measured exit is a taker stop-loss (SL fills are")
    w("deterministic in both models); the sensitivity row only diverges if maker TP fills are ever decisive.*")
    w("")
    w(f"Vetoed debate accounting: {res['n_vetoed']} vetoes → {v['reconstructed']} setups reconstructed, "
      f"deduplicated to {v['unique_setups']} unique (symbol, entry candle, side) — {v['reconstructed'] - v['unique_setups']} "
      f"repeat deliberations of the same live signal dropped from the statistics — {v['n_measured']} replayed to a TP/SL/edge outcome. "
      f"Skipped: " + ", ".join(f"{k}={n}" for k, n in res["skipped"].items()) + ".")
    w("")
    w("Shadow exit-reason distribution (touch): " + (", ".join(f"{k}={n}" for k, n in v["exit_reasons"].items()) or "none"))
    w("")
    w("## 3. Boardroom value (difference of two measured populations)")
    w("")
    if bv:
        excl = bv["lower"] > 0
        w(f"Approved realized EV − Combined EV = **{money(bv['diff'])}/trade**, bootstrap 90% CI "
          f"[{bv['lower']:+.3f}, {bv['upper']:+.3f}] (N approved={bv['n_a']}, N combined={bv['n_b']}).")
        if excl:
            w("The CI excludes zero: at current sample the Boardroom's vetoes are measurably positive-value filtering.")
        else:
            w("**The CI straddles zero: with the current sample size the Boardroom's filtering value is statistically "
              "indistinguishable from zero.** No 'capital defended' claim is supportable.")
    else:
        w("NOT MEASURABLE — one or both populations have N < 2; bootstrap CI undefined.")
    w("")
    w("## 4. Conviction calibration (executed population only)")
    w("")
    w("| Conviction bin | Deliberations | Matched executed | Win rate | EV/trade | Total |")
    w("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for row in res["calibration"]:
        p = row["pop"]
        wr = ('%.1f%%' % p['win_rate']) if p['win_rate'] is not None else "NOT MEASURABLE"
        w(f"| {row['bin']} | {row['deliberations']} | {row['matched']} | {wr} | {money(p['ev'])} | {money(p['total_pnl'])} |")
    w("")
    w("> Bin sizes are small; these are descriptive counts, not inferential results.")
    w("")
    w("## 5. Honest caveats")
    w("")
    w("1. **Shadow, not realized:** vetoed setups were never live; replay fills assume touch execution (upper bound) with 1-tick shown for sensitivity.")
    w("2. **Holding model:** swing exits beyond TP/SL (trailing/structural/time exits observed in the live ledger) are NOT modeled; shadow PnL reflects the bracket-only geometry.")
    w("3. **AI exit geometry:** live entries can adopt structure-calibrated SL/TP (confidence ≥ 0.70) which is not reconstructable from debate rows — baseline ATR geometry used throughout (labeled assumption).")
    w("4. **Leverage tier:** standard 3× assumed for all shadow trades; premium 5× requires conviction ≥ 0.85 AND ADX ≥ 32 (unavailable for vetoed rows).")
    w("5. **Duplicate vetoes:** the boardroom re-deliberates the same live signal every few minutes; shadow statistics deduplicate to unique (symbol, entry candle, side) setups, but setups on the same market day remain correlated and the iid bootstrap does not correct for that dependence.")
    w("6. **Provenance paused:** until `served_model_id` accumulates (FIX 2), no training-cutoff claims are made.")
    w("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report generated at {out_path}")

    if dump_csv:
        pd.DataFrame(res["shadow_rows"]).to_csv(dump_csv, index=False)
        print(f"[OK] Shadow trade rows dumped to {dump_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Boardroom counterfactuals via real veto shadow replay")
    parser.add_argument("--ledger-db", type=str, default="trading_ledger.db")
    parser.add_argument("--data-db", type=str, default="trading_data.db")
    parser.add_argument("--cache-dir", type=str, default=os.path.join("data", "counterfactual"))
    parser.add_argument("--out", type=str, default="audits/boardroom_counterfactual_report_2026-10-13.md")
    parser.add_argument("--dump-csv", type=str, default="audits/boardroom_shadow_trades_2026-10-13.csv")
    args = parser.parse_args()

    res = evaluate(args.ledger_db, args.data_db, args.cache_dir)
    generate_report(res, args.out, args.dump_csv)
    v = res["vetoed_shadow"]
    print(json.dumps({
        "n_vetoed": res["n_vetoed"], "reconstructed": v["reconstructed"], "measured": v["n_measured"],
        "shadow_ev": v["ev"], "shadow_ci": (v["ci"] or {}).get("lower"), 
        "approved_ev": res["approved_realized"]["ev"],
        "boardroom_value": res["boardroom_value"],
    }, default=str, indent=2))


if __name__ == "__main__":
    main()
