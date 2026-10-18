# nightly_auditor.py — Agent 5: The Nightly Auditor / The Teacher
# MARL Trading Swarm | Post-Trade Forensic Engine & Reward Calculator
#
# Primary DB  : trading_data.db   → table: trades  (live production ledger)
# Fallback DB : audit_sandbox.db  → table: trades  (mock test context)
# Output      : learned_memory.json  (appended each run)

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from typing import Optional

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] NIGHTLY_AUDITOR: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nightly_auditor")

# ── Absolute Paths ─────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT       = os.path.dirname(BASE_DIR)
LIVE_DB_PATH       = os.path.join(PROJECT_ROOT, "trading_data.db") if os.path.exists(os.path.join(PROJECT_ROOT, "trading_data.db")) else os.path.join(BASE_DIR, "trading_data.db")
SANDBOX_DB_PATH    = os.path.join(BASE_DIR, "audit_sandbox.db")
MEMORY_PATH        = os.path.join(PROJECT_ROOT, "data", "state", "learned_memory.json") if os.path.exists(os.path.join(PROJECT_ROOT, "data", "state", "learned_memory.json")) else os.path.join(BASE_DIR, "learned_memory.json")


# ── Reward Hyperparameters ────────────────────────────────────────────────────
MU                 = 0.001   # Time-exposure penalty coefficient (per minute)
WIN_RATE_THRESHOLD = 0.45    # Below this → defensive rule triggers
DEFAULT_DURATION   = 30.0    # Assumed duration (mins) if exposure_time_mins is NULL


# ── DB Context Manager ────────────────────────────────────────────────────────
@contextmanager
def get_db(path: str, timeout: float = 10.0):
    """
    Safe SQLite connection context manager.
    Enforces WAL mode and busy_timeout to prevent file locks
    in a multi-process swarm environment.
    """
    conn = sqlite3.connect(path, timeout=timeout)
    conn.row_factory = sqlite3.Row          # Named column access
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=10000;")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Reward Formula ────────────────────────────────────────────────────────────
def compute_reward(pnl: float, duration_mins: float, mu: float = MU) -> float:
    """
    Risk-adjusted RL Reward Signal:
        R = PnL - (mu * duration_mins)

    Penalises capital stagnation: a trade that ties up capital for a long
    time must generate proportionally more PnL to achieve the same reward
    as a short, decisive trade.
    """
    return round(pnl - (mu * duration_mins), 6)


# ── NightlyAuditor ────────────────────────────────────────────────────────────
class NightlyAuditor:
    """
    Agent 5 — The Nightly Auditor / The Teacher.

    Workflow:
      1. Ingest closed trades from the last 24 h.
      2. Compute a risk-adjusted reward signal R for each trade.
      3. Aggregate win/loss metrics; fire defensive rule if win_rate < 45%.
      4. Persist reward logs + rules into learned_memory.json.
    """

    def __init__(
        self,
        db_path:     str = LIVE_DB_PATH,
        memory_path: str = MEMORY_PATH,
        mu:          float = MU,
    ):
        self.db_path     = db_path
        self.memory_path = memory_path
        self.mu          = mu
        log.info(f"NightlyAuditor initialised. DB={os.path.basename(db_path)}  mu={mu}")

    # ── 1. Database Ingestion ─────────────────────────────────────────────────
    def fetch_closed_trades(self, lookback_hours: int = 24) -> list[dict]:
        """
        Pulls all closed trades timestamped within the last `lookback_hours`
        from the `trades` table. Returns a list of row dicts.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        ).isoformat()

        query = """
            SELECT
                id,
                timestamp,
                symbol,
                side,
                pnl,
                exposure_time_mins,
                regime,
                notes,
                ai_rationale,
                quality
            FROM trades
            WHERE (status = 'closed' OR status = 'CLOSED')
              AND timestamp >= ?
            ORDER BY timestamp ASC
        """

        rows = []
        try:
            with get_db(self.db_path) as conn:
                cursor = conn.execute(query, (cutoff,))
                rows   = [dict(r) for r in cursor.fetchall()]
            log.info(f"Ingested {len(rows)} closed trade(s) from the last {lookback_hours}h.")
        except sqlite3.OperationalError as e:
            log.warning(f"DB query failed ({e}). Returning empty trade list.")
        except Exception as e:
            log.error(f"Unexpected DB error: {e}")

        return rows

    # ── 2. Reward Signal Calculation ──────────────────────────────────────────
    def calculate_rewards(self, trades: list[dict]) -> list[dict]:
        """
        Enriches each trade dict with:
          - duration_mins  : coerced exposure time
          - reward         : R = PnL - (mu * duration_mins)
          - outcome        : 'WIN' | 'LOSS' | 'BREAKEVEN'
        """
        enriched = []
        for t in trades:
            pnl      = float(t.get("pnl") or 0.0)
            dur      = float(t.get("exposure_time_mins") or DEFAULT_DURATION)
            reward   = compute_reward(pnl, dur, self.mu)

            if pnl > 0:
                outcome = "WIN"
            elif pnl < 0:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"

            enriched.append({
                **t,
                "duration_mins": round(dur, 2),
                "reward":        reward,
                "outcome":       outcome,
            })

            log.info(
                f"  Trade #{t.get('id','?')} | {t.get('symbol','?')} {t.get('side','?').upper()} "
                f"| PnL={pnl:+.4f}  Dur={dur:.1f}m  R={reward:+.6f}  [{outcome}]"
            )

        return enriched

    # ── 3. Self-Reflection & Rule Generation ──────────────────────────────────
    def generate_rules(self, enriched_trades: list[dict]) -> dict:
        """
        Aggregates session metrics and emits defensive rules when
        win rate falls below WIN_RATE_THRESHOLD.

        Returns a metrics dict containing:
          total, wins, losses, win_rate, total_pnl,
          total_reward, avg_reward, rules[]
        """
        total   = len(enriched_trades)
        wins    = sum(1 for t in enriched_trades if t["outcome"] == "WIN")
        losses  = sum(1 for t in enriched_trades if t["outcome"] == "LOSS")
        win_rate = (wins / total) if total > 0 else 0.0

        total_pnl    = sum(t.get("pnl", 0.0)    for t in enriched_trades)
        total_reward = sum(t.get("reward", 0.0)  for t in enriched_trades)
        avg_reward   = (total_reward / total) if total > 0 else 0.0

        rules = []

        # Rule 1: Low win-rate defence
        if win_rate < WIN_RATE_THRESHOLD:
            rule = (
                f"Market conditions unstable (win_rate={win_rate:.1%} < "
                f"{WIN_RATE_THRESHOLD:.0%}); increase entry confidence thresholds."
            )
            rules.append(rule)
            log.warning(f"[RULE GENERATED] {rule}")

        # Rule 2: Negative total reward defence
        if total_reward < 0:
            rule = (
                f"Negative session reward ({total_reward:+.4f}); "
                f"review duration discipline — reduce max hold time."
            )
            rules.append(rule)
            log.warning(f"[RULE GENERATED] {rule}")

        # Rule 3: Zero-trade guard
        if total == 0:
            rules.append("No trades executed in the last 24h. Verify signal pipeline health.")
            log.warning("[RULE GENERATED] No trades found in audit window.")

        metrics = {
            "total_trades":  total,
            "wins":          wins,
            "losses":        losses,
            "win_rate":      round(win_rate, 4),
            "total_pnl":     round(total_pnl, 6),
            "total_reward":  round(total_reward, 6),
            "avg_reward":    round(avg_reward, 6),
            "rules":         rules,
        }

        log.info(
            f"Session metrics: {wins}W/{losses}L  "
            f"WinRate={win_rate:.1%}  TotalPnL={total_pnl:+.4f}  "
            f"TotalReward={total_reward:+.6f}  Rules={len(rules)}"
        )
        return metrics

    # ── 4. State Persistence ──────────────────────────────────────────────────
    def persist_to_memory(self, metrics: dict, enriched_trades: list[dict]) -> None:
        """
        Loads existing learned_memory.json (or creates a fresh file),
        appends the current audit session, and writes back atomically.
        """
        # Load existing memory
        memory: dict = {}
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r", encoding="utf-8") as f:
                    memory = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log.warning(f"Could not read existing memory file ({e}). Starting fresh.")

        # Append current session entry
        session_key = f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        memory[session_key] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metrics":       metrics,
            "reward_log":    [
                {
                    "id":            t.get("id"),
                    "symbol":        t.get("symbol"),
                    "side":          t.get("side"),
                    "pnl":           t.get("pnl"),
                    "duration_mins": t.get("duration_mins"),
                    "reward":        t.get("reward"),
                    "outcome":       t.get("outcome"),
                    "regime":        t.get("regime"),
                }
                for t in enriched_trades
            ],
            "rules": metrics.get("rules", []),
        }

        # Atomic write via temp file swap
        tmp_path = self.memory_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(memory, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.memory_path)
            log.info(f"Session '{session_key}' persisted to {self.memory_path}")
        except IOError as e:
            log.error(f"Failed to write learned_memory.json: {e}")

    # ── Full Audit Pipeline ───────────────────────────────────────────────────
    def run(self, lookback_hours: int = 24) -> dict:
        """
        Execute the full audit workflow and return the session metrics dict.
        """
        log.info("=" * 55)
        log.info("  Starting Nightly Architecture Audit (V13.0)...")
        log.info("=" * 55)

        trades          = self.fetch_closed_trades(lookback_hours)
        enriched        = self.calculate_rewards(trades)
        metrics         = self.generate_rules(enriched)
        self.persist_to_memory(metrics, enriched)

        log.info("Nightly audit complete.")
        return metrics


# ── Mock DB Seed Helper ───────────────────────────────────────────────────────
def _seed_mock_db(db_path: str) -> None:
    """
    Creates a minimal `trades` table in a sandbox DB and inserts 3 sample trades:
      Trade A — Quick WIN:      +2.50 USDT, 15 min exposure
      Trade B — Moderate WIN:   +0.80 USDT, 45 min exposure
      Trade C — Long LOSS:      -1.20 USDT, 210 min exposure  (capital stagnation test)
    """
    now     = datetime.now(timezone.utc)
    fmt     = lambda h, m: (now - timedelta(hours=h, minutes=m)).isoformat()

    schema = """
        CREATE TABLE IF NOT EXISTS trades (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT NOT NULL,
            symbol            TEXT NOT NULL,
            side              TEXT NOT NULL,
            type              TEXT DEFAULT 'unknown',
            pnl               REAL,
            exposure_time_mins REAL,
            regime            TEXT,
            notes             TEXT,
            ai_rationale      TEXT,
            quality           REAL DEFAULT 0.0,
            status            TEXT DEFAULT 'closed'
        );
    """

    sample_trades = [
        # (timestamp, symbol, side, type, pnl, exposure_mins, regime, notes)
        (fmt(2, 0),  "BTC/USDT", "long",  "tp_hit",  2.50,  15.0,  "TRENDING", "Quick breakout capture. EMA stack confirmed."),
        (fmt(5, 0),  "ETH/USDT", "short", "sl_hit",  0.80,  45.0,  "RANGING",  "Moderate win on range fade."),
        (fmt(10, 0), "SOL/USDT", "long",  "sl_hit",  -1.20, 210.0, "CHOPPY",   "Extended hold into chop. Duration penalty applies."),
    ]

    with get_db(db_path) as conn:
        conn.execute(schema)
        conn.executemany(
            """INSERT INTO trades
               (timestamp, symbol, side, type, pnl, exposure_time_mins, regime, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            sample_trades,
        )
    log.info(f"Mock DB seeded with {len(sample_trades)} sample trades at {db_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    log.info("─" * 55)
    log.info("  NightlyAuditor — Mock End-to-End Self-Test")
    log.info("─" * 55)

    # Use a throw-away sandbox DB so the live ledger is never touched
    _seed_mock_db(SANDBOX_DB_PATH)

    auditor = NightlyAuditor(
        db_path     = SANDBOX_DB_PATH,
        memory_path = MEMORY_PATH,
        mu          = MU,
    )

    metrics = auditor.run(lookback_hours=24)

    # ── Pretty-print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  NIGHTLY AUDIT — SESSION SUMMARY")
    print("=" * 55)
    print(f"  Total Trades  : {metrics['total_trades']}")
    print(f"  Wins / Losses : {metrics['wins']}W  /  {metrics['losses']}L")
    print(f"  Win Rate      : {metrics['win_rate']:.1%}")
    print(f"  Total PnL     : {metrics['total_pnl']:+.4f} USDT")
    print(f"  Total Reward  : {metrics['total_reward']:+.6f}  (R = PnL - mu*Duration)")
    print(f"  Avg Reward    : {metrics['avg_reward']:+.6f}")
    print(f"  Rules Fired   : {len(metrics['rules'])}")
    for i, rule in enumerate(metrics["rules"], 1):
        print(f"    Rule {i}: {rule}")
    print("=" * 55)
    print(f"  Memory File   : {MEMORY_PATH}")
    print("=" * 55 + "\n")

    # Verify the JSON was written
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
        # Only inspect audit_ session keys written by this script
        audit_keys = sorted(k for k in blob.keys() if k.startswith("audit_"))
        if audit_keys:
            last_key = audit_keys[-1]
            session  = blob[last_key]
            print(f"  [VERIFY] Last audit session key : '{last_key}'")
            print(f"  [VERIFY] Rules stored           : {session.get('rules', [])}")
            print(f"  [VERIFY] Reward log entries     : {len(session.get('reward_log', []))}")
            print(f"  [VERIFY] Win rate persisted     : {session['metrics']['win_rate']:.1%}")
        else:
            print("  [VERIFY] No audit_ session keys found yet.")
        print()

    sys.exit(0)
