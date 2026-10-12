"""
economic_calendar.py — High-Impact Macroeconomic Event & News Calendar Gate
Tracks major market-moving macroeconomic events (FOMC, CPI, NFP, PPI) and
suppresses entry signals within a ±30-minute high-volatility window.
"""

import os
import sys
import time
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] ECON_CALENDAR: %(message)s")
log = logging.getLogger("econ_calendar")

HIGH_IMPACT_EVENTS_2026 = [
    ("2026-08-12", "12:30", "US CPI Inflation YoY/MoM", "HIGH"),
    ("2026-08-13", "12:30", "US PPI Final Demand MoM", "HIGH"),
    ("2026-08-19", "18:00", "FOMC Meeting Minutes Release", "HIGH"),
    ("2026-08-21", "14:00", "Jackson Hole Economic Symposium Opening", "HIGH"),
    ("2026-08-27", "12:30", "US GDP Q2 Second Estimate", "HIGH"),
    ("2026-08-28", "12:30", "US Core PCE Price Index MoM", "HIGH"),
    ("2026-09-04", "12:30", "US Non-Farm Payrolls & Unemployment Rate", "HIGH"),
    ("2026-09-11", "12:30", "US CPI Inflation YoY/MoM", "HIGH"),
    ("2026-09-16", "18:00", "FOMC Interest Rate Decision & Fed Press Conf", "CRITICAL"),
    ("2026-09-25", "12:30", "US Core PCE Price Index MoM", "HIGH"),
    ("2026-10-02", "12:30", "US Non-Farm Payrolls & Unemployment Rate", "HIGH"),
    ("2026-10-13", "12:30", "US CPI Inflation YoY/MoM", "HIGH"),
    ("2026-10-28", "18:00", "FOMC Interest Rate Decision & Fed Press Conf", "CRITICAL"),
    ("2026-11-06", "13:30", "US Non-Farm Payrolls", "HIGH"),
    ("2026-11-12", "13:30", "US CPI Inflation YoY/MoM", "HIGH"),
    ("2026-11-25", "13:30", "US Core PCE Price Index MoM", "HIGH"),
    ("2026-12-04", "13:30", "US Non-Farm Payrolls", "HIGH"),
    ("2026-12-10", "13:30", "US CPI Inflation YoY/MoM", "HIGH"),
    ("2026-12-16", "19:00", "FOMC Interest Rate Decision & Summary of Economic Projections", "CRITICAL"),
]

class EconomicCalendarGate:
    def __init__(self, db_path="trading_data.db", pre_window_mins: int = None, post_window_mins: int = None, window_minutes: int = None):
        self.db_path = db_path
        import config
        # Asymmetric Macro Shield: 60m pre-event lockout, 120m post-event lockout (covers NY open flushes)
        self.pre_window_mins = pre_window_mins if pre_window_mins is not None else int(getattr(config, "PRE_NEWS_MINUTES", 60))
        self.post_window_mins = post_window_mins if post_window_mins is not None else int(getattr(config, "POST_NEWS_MINUTES", 120))
        if window_minutes is not None:
            self.pre_window_mins = window_minutes
            self.post_window_mins = window_minutes
        self.window_minutes = max(self.pre_window_mins, self.post_window_mins)
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS macro_news_calendar (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_date TEXT,
                        event_time_utc TEXT,
                        event_name TEXT,
                        impact TEXT,
                        is_active INTEGER DEFAULT 0
                    )
                """)
                count = conn.execute("SELECT COUNT(*) FROM macro_news_calendar").fetchone()[0]
                if count == 0:
                    for (ev_d, ev_t, ev_n, ev_i) in HIGH_IMPACT_EVENTS_2026:
                        conn.execute("""
                            INSERT INTO macro_news_calendar (event_date, event_time_utc, event_name, impact)
                            VALUES (?, ?, ?, ?)
                        """, (ev_d, ev_t, ev_n, ev_i))
                    conn.commit()
        except Exception as e:
            log.warning(f"DB init note in economic_calendar: {e}")

    def check_event_window(self, target_time: datetime = None) -> tuple[bool, str, float]:
        now = target_time or datetime.now(timezone.utc)
        
        for (ev_d, ev_t, ev_n, ev_i) in HIGH_IMPACT_EVENTS_2026:
            try:
                ev_str = f"{ev_d} {ev_t}:00"
                ev_dt = datetime.strptime(ev_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                diff_secs = (now - ev_dt).total_seconds()
                diff_mins = diff_secs / 60.0

                # Asymmetric Shield:
                # If before event (diff_mins < 0): mute if within pre_window_mins (60m)
                # If after event (diff_mins >= 0): mute if within post_window_mins (120m)
                is_active = False
                if diff_mins < 0 and abs(diff_mins) <= self.pre_window_mins:
                    is_active = True
                elif diff_mins >= 0 and diff_mins <= self.post_window_mins:
                    is_active = True

                if is_active:
                    timing_str = f"in {abs(diff_mins):.0f}m" if diff_mins < 0 else f"{diff_mins:.0f}m ago (Cooldown active)"
                    reason = f"[{ev_i}] {ev_n} ({timing_str})"
                    return True, reason, diff_mins
            except Exception:
                continue

        next_event_str = "None in next 7 days"
        min_future_diff = float("inf")
        for (ev_d, ev_t, ev_n, ev_i) in HIGH_IMPACT_EVENTS_2026:
            try:
                ev_str = f"{ev_d} {ev_t}:00"
                ev_dt = datetime.strptime(ev_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                diff_secs = (ev_dt - now).total_seconds()
                if 0 <= diff_secs < min_future_diff:
                    min_future_diff = diff_secs
                    hours = diff_secs / 3600.0
                    next_event_str = f"{ev_n} on {ev_d} {ev_t} UTC (in {hours:.1f}h)"
            except Exception:
                continue

        return False, next_event_str, min_future_diff / 60.0 if min_future_diff < float("inf") else 9999.0

    def update_system_state(self):
        is_active, reason, _ = self.check_event_window()
        ts = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('news_event_active', ?, ?)
                """, ("1" if is_active else "0", ts))
                conn.execute("""
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES ('news_event_reason', ?, ?)
                """, (reason, ts))
                conn.commit()
            if is_active:
                log.warning(f"[NEWS-GATE-ACTIVE] 🛑 High-impact event active: {reason}. Trade entries suppressed.")
            else:
                log.info(f"[NEWS-GATE-CLEAR] 🟢 No active news events. Next: {reason}")
        except Exception as e:
            log.error(f"Failed to update economic calendar system_state: {e}")

if __name__ == "__main__":
    gate = EconomicCalendarGate()
    is_active, info, diff = gate.check_event_window()
    gate.update_system_state()
    print(f"Active Event Window: {is_active}")
    print(f"Details / Next Event: {info}")
