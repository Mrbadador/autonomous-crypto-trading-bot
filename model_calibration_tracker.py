"""
model_calibration_tracker.py — AI Signal Calibration & Brier Score Tracking Engine
Monitors AI prediction confidence vs actual win rates to detect calibration drift
and provide empirical calibration curves for conviction gating.
"""

import os
import sys
import json
import sqlite3
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] CALIBRATION: %(message)s")
log = logging.getLogger("calibration")

class ModelCalibrationTracker:
    def __init__(self, db_ledger="trading_ledger.db", db_data="trading_data.db"):
        self.db_ledger = db_ledger
        self.db_data = db_data
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_ledger, timeout=10.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_calibration_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id INTEGER,
                        symbol TEXT,
                        signal_source TEXT,
                        predicted_confidence REAL,
                        predicted_action TEXT,
                        actual_pnl REAL,
                        is_win INTEGER,
                        created_at TEXT,
                        closed_at TEXT
                    )
                """)
        except Exception as e:
            log.warning(f"DB init note in calibration tracker: {e}")

    def log_prediction_outcome(self, trade_id: int, symbol: str, signal_source: str,
                               confidence: float, action: str, pnl: float,
                               created_at: str = None, closed_at: str = None):
        """Records an individual prediction vs realized outcome event."""
        # Normalize confidence to 0.0 - 1.0 range
        conf_norm = confidence / 100.0 if confidence > 1.0 else confidence
        conf_norm = max(0.0, min(1.0, conf_norm))
        is_win = 1 if pnl > 0 else 0
        now_ts = datetime.now(timezone.utc).isoformat()
        
        try:
            with sqlite3.connect(self.db_ledger, timeout=10.0) as conn:
                conn.execute("""
                    INSERT INTO model_calibration_log 
                    (trade_id, symbol, signal_source, predicted_confidence, predicted_action, actual_pnl, is_win, created_at, closed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade_id, symbol, signal_source, conf_norm, action, pnl, is_win, created_at or now_ts, closed_at or now_ts))
                conn.commit()
            log.info(f"[CALIBRATION LOGGED] Trade #{trade_id} {symbol} {action} | PredConf={conf_norm*100:.1f}% | Win={is_win} (PnL=${pnl:+.2f})")
        except Exception as e:
            log.error(f"Failed to log calibration record: {e}")

    def bootstrap_from_history(self):
        """Populates calibration table from existing historical closed trades if empty."""
        try:
            with sqlite3.connect(self.db_ledger, timeout=5.0) as conn_ledger:
                cnt = conn_ledger.execute("SELECT COUNT(*) FROM model_calibration_log").fetchone()[0]
                if cnt > 0:
                    log.info(f"Calibration table already contains {cnt} records. Skipping bootstrap.")
                    return

            with sqlite3.connect(self.db_data, timeout=5.0) as conn_data:
                conn_data.row_factory = sqlite3.Row
                rows = conn_data.execute("""
                    SELECT id, timestamp, symbol, side, pnl, quality, ai_rationale 
                    FROM trades 
                    WHERE (status = 'closed' OR status = 'CLOSED')
                """).fetchall()

            if not rows:
                return

            log.info(f"Bootstrapping calibration log with {len(rows)} historical trades...")
            with sqlite3.connect(self.db_ledger, timeout=10.0) as conn_ledger:
                for r in rows:
                    pnl = float(r["pnl"] or 0.0)
                    is_win = 1 if pnl > 0 else 0
                    q = float(r["quality"] or 0.70)
                    conf = q / 100.0 if q > 1.0 else q
                    conn_ledger.execute("""
                        INSERT INTO model_calibration_log 
                        (trade_id, symbol, signal_source, predicted_confidence, predicted_action, actual_pnl, is_win, created_at, closed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (r["id"], r["symbol"], "AI_Boardroom", conf, r["side"], pnl, is_win, r["timestamp"], r["timestamp"]))
                conn_ledger.commit()
            log.info(f"Successfully bootstrapped {len(rows)} historical trade calibration records.")
        except Exception as e:
            log.error(f"Failed to bootstrap calibration log: {e}")

    def compute_calibration_metrics(self) -> dict:
        """Calculates Brier Score, Reliability Bins, and Calibration Error."""
        self.bootstrap_from_history()
        
        try:
            with sqlite3.connect(self.db_ledger, timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT predicted_confidence, is_win, actual_pnl FROM model_calibration_log").fetchall()
        except Exception as e:
            log.error(f"Failed to read calibration data: {e}")
            return {}

        if not rows:
            return {"total_samples": 0, "brier_score": 0.0, "bins": []}

        n = len(rows)
        brier_sum = 0.0
        
        # Reliability bins: [0.50-0.60, 0.60-0.70, 0.70-0.80, 0.80-0.90, 0.90-1.00]
        bin_edges = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
        bins_data = {i: {"count": 0, "wins": 0, "conf_sum": 0.0} for i in range(len(bin_edges)-1)}

        for r in rows:
            p = float(r["predicted_confidence"])
            o = int(r["is_win"])
            brier_sum += (p - o) ** 2

            for i in range(len(bin_edges) - 1):
                low, high = bin_edges[i], bin_edges[i+1]
                if (i == len(bin_edges)-2 and low <= p <= high) or (low <= p < high):
                    bins_data[i]["count"] += 1
                    bins_data[i]["wins"] += o
                    bins_data[i]["conf_sum"] += p
                    break

        brier_score = brier_sum / n
        
        bins_summary = []
        for i, b in bins_data.items():
            if b["count"] > 0:
                avg_conf = b["conf_sum"] / b["count"]
                obs_wr = (b["wins"] / b["count"]) * 100.0
                bins_summary.append({
                    "bin_range": f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%",
                    "sample_count": b["count"],
                    "avg_predicted_conf": round(avg_conf * 100.0, 1),
                    "observed_win_rate": round(obs_wr, 1),
                    "calibration_gap": round(obs_wr - (avg_conf * 100.0), 1)
                })

        overall_wr = sum(r["is_win"] for r in rows) / n * 100.0

        return {
            "total_samples": n,
            "overall_win_rate": round(overall_wr, 1),
            "brier_score": round(brier_score, 4),
            "calibration_grade": "EXCELLENT" if brier_score < 0.18 else "GOOD" if brier_score < 0.25 else "NEEDS_TUNING",
            "bins": bins_summary
        }

if __name__ == "__main__":
    tracker = ModelCalibrationTracker()
    metrics = tracker.compute_calibration_metrics()

    # An empty calibration log is an absence, not a measured zero: the tracker returns
    # total_samples=0 with no overall_win_rate and no calibration_grade (and a brier_score of
    # 0.0 standing in for a score with no samples to average over). Printing 0% and 0.0 there
    # claims a measurement nobody took, so the report says what it does not have instead.
    samples = metrics.get("total_samples")
    try:
        samples = int(samples)
    except (TypeError, ValueError):
        samples = None
    measured = bool(samples) and "overall_win_rate" in metrics
    if samples is None:
        absent_reason = "calibration metrics unavailable"
    elif samples == 0:
        absent_reason = "no signals evaluated yet"
    else:
        absent_reason = "not measured"
    print("\n=== AI MODEL CALIBRATION & RELIABILITY REPORT ===")
    print(f"Total Evaluated Signals: {samples if samples is not None else 'n/a'}")
    if measured:
        print(f"Overall Historical Win Rate: {metrics.get('overall_win_rate')}%")
        print(f"Brier Calibration Score: {metrics.get('brier_score')} (Grade: {metrics.get('calibration_grade', 'not measured')})")
    else:
        print(f"Overall Historical Win Rate: n/a ({absent_reason})")
        print(f"Brier Calibration Score: n/a ({absent_reason}) (Grade: not measured)\n")
    print("Reliability Curve Bins:")
    print(f"{'Bin':<12} | {'Samples':<8} | {'Pred Conf':<10} | {'Obs Win Rate':<12} | {'Calib Gap':<10}")
    print("-" * 60)
    for b in metrics.get("bins", []):
        print(f"{b['bin_range']:<12} | {b['sample_count']:<8} | {b['avg_predicted_conf']:<9}% | {b['observed_win_rate']:<11}% | {b['calibration_gap']:+<9.1f}%")
