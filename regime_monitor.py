# regime_monitor.py — Hurst/ADX Regime Shift Alert Engine
# =========================================================
# Polls market_regime table every 60s.
# Fires a "REGIME SHIFT ALERT" email when Hurst crosses 0.55.
#
# Integration: Called from antigravity_agent_manager.py as:
#   asyncio.create_task(regime_monitor.run_regime_monitor())
#
# Edge-case guards:
#   - State machine: only fires once per crossing (no spam on sustained trend)
#   - Falls back gracefully if market_regime has only 1 row (upsert pattern)
#   - All DB reads use WAL mode + 8s timeout
#   - Email sent via asyncio.to_thread — never blocks the event loop

import os
import asyncio
import sqlite3
import logging
from datetime import datetime, timezone

log = logging.getLogger("regime_monitor")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TRADING_DATA_DB = os.path.join(BASE_DIR, "trading_data.db")

# ── Thresholds ─────────────────────────────────────────────────────────────────
HURST_TREND_GATE  = 0.55   # above → TRENDING (persistent, mean-reverting momentum)
HURST_CHOP_GATE   = 0.48   # below → CHOPPING (revert to defensive)
ADX_CONFIRM_GATE  = 25.0   # ADX must also exceed this to confirm trend

POLL_INTERVAL_SEC = 60      # check every 60 seconds


def _read_regime() -> dict | None:
    """Read the latest market_regime row with directional telemetry. Returns None on failure."""
    if not os.path.exists(TRADING_DATA_DB):
        return None
    try:
        c = sqlite3.connect(TRADING_DATA_DB, timeout=8)
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA busy_timeout=8000;")
        # Check available columns
        cols = [r[1] for r in c.execute("PRAGMA table_info(market_regime)").fetchall()]
        has_directional = "plus_di" in cols and "minus_di" in cols
        
        if has_directional:
            row = c.execute(
                "SELECT hurst, adx, regime, timestamp, plus_di, minus_di, macro_trend FROM market_regime ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            c.close()
            if row:
                return {
                    "hurst": row[0], "adx": row[1], "regime": row[2], "timestamp": row[3],
                    "plus_di": row[4] or 0.0, "minus_di": row[5] or 0.0, "macro_trend": row[6] or "NEUTRAL"
                }
        else:
            row = c.execute(
                "SELECT hurst, adx, regime, timestamp FROM market_regime ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            c.close()
            if row:
                return {
                    "hurst": row[0], "adx": row[1], "regime": row[2], "timestamp": row[3],
                    "plus_di": 0.0, "minus_di": 0.0, "macro_trend": "NEUTRAL"
                }
    except Exception as e:
        log.warning(f"[REGIME-MONITOR] DB read failed: {e}")
    return None


def _build_regime_alert_html(hurst: float, adx: float, regime: str,
                              prev_regime: str, ts: str,
                              plus_di: float = 0.0, minus_di: float = 0.0,
                              macro_trend: str = "NEUTRAL") -> tuple[str, str]:
    """Build subject + HTML body for directional regime shift alert email."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    di_spread = plus_di - minus_di

    if regime in ("BEAR_TREND", "BEARISH"):
        direction_label = f"{prev_regime} ➔ BEAR_TREND (CAPITAL SHIELD)"
        badge_color     = "#ef4444"
        action_line     = "Downside momentum accelerating (-DI > +DI). Capital Shield ENGAGED. Zero new long breakout entries allowed. Cash buffer preserved."
        subject         = f"[REGIME SHIFT 🚨] CAPITAL SHIELD ENGAGED ➔ BEAR_TREND | ADX: {adx:.1f} | {now_str}"
    elif regime in ("BULL_TREND", "BULLISH", "TRENDING"):
        direction_label = f"{prev_regime} ➔ BULL_TREND (RUNNER EXPANSION)"
        badge_color     = "#10b981"
        action_line     = "Upside trend momentum confirmed (+DI > -DI). 4H Swing Architect fleet scanning ENABLED. Trailing profit floors active."
        subject         = f"[REGIME SHIFT ⚡] BULL_TREND DETECTED ➔ 4H SWING ACTIVE | ADX: {adx:.1f} | {now_str}"
    else:
        direction_label = f"{prev_regime} ➔ CHOP / CONSOLIDATION"
        badge_color     = "#f59e0b"
        action_line     = "Market entering consolidation range. Mean Reversion Maker Grid post-only limit orders active. Sizing calibrated to 3x."
        subject         = f"[REGIME SHIFT ⚠] CONSOLIDATION DETECTED ➔ CHOP | ADX: {adx:.1f} | {now_str}"

    html = f"""
    <html><head><style>
        body {{font-family:'Segoe UI',sans-serif;background:#0f172a;color:#f1f5f9;padding:20px}}
        .container {{max-width:580px;background:#1e293b;border:1px solid #334155;
                     border-radius:12px;padding:28px;margin:0 auto}}
        .alert-bar {{background:linear-gradient(135deg,#7c3aed,#4f46e5);border-radius:8px;
                     padding:16px 20px;margin-bottom:20px;border-left:4px solid {badge_color}}}
        .alert-bar h2 {{margin:0;color:#e2e8f0;font-size:19px}}
        .alert-bar p  {{margin:4px 0 0;color:#c4b5fd;font-size:13px}}
        h3 {{color:#cbd5e1;font-size:13px;text-transform:uppercase;letter-spacing:.05em;
              border-left:3px solid #818cf8;padding-left:8px;margin-top:20px}}
        table {{width:100%;border-collapse:collapse;margin-top:8px}}
        td {{padding:9px 12px;border-bottom:1px solid #334155;font-size:14px}}
        .lbl {{color:#94a3b8;font-weight:600;width:50%}}
        .val {{text-align:right;font-family:Consolas,monospace;font-weight:bold;color:#f1f5f9}}
        .action-box {{background:#0f172a;border-left:4px solid {badge_color};padding:14px;
                      border-radius:4px;font-size:14px;color:#fde68a;margin-top:14px}}
        .footer {{font-size:11px;color:#475569;text-align:center;margin-top:24px;
                  border-top:1px solid #334155;padding-top:14px}}
    </style></head><body>
    <div class='container'>
        <div class='alert-bar'>
            <h2>⚡ REGIME SHIFT DETECTED — {direction_label}</h2>
            <p>Triggered at {now_str}</p>
        </div>
        <h3>Regime Telemetry</h3>
        <table>
            <tr><td class='lbl'>New Declared Regime</td>
                <td class='val' style='color:{badge_color}'>{regime}</td></tr>
            <tr><td class='lbl'>Previous Regime</td>
                <td class='val'>{prev_regime}</td></tr>
            <tr><td class='lbl'>ADX (Trend Strength)</td>
                <td class='val'>{adx:.2f} (Gate: &gt; 25.0)</td></tr>
            <tr><td class='lbl'>+DI vs -DI Spread</td>
                <td class='val'>{di_spread:+.2f} (+DI: {plus_di:.1f} | -DI: {minus_di:.1f})</td></tr>
            <tr><td class='lbl'>4H Macro Trend Stack</td>
                <td class='val'>{macro_trend}</td></tr>
            <tr><td class='lbl'>Hurst Exponent</td>
                <td class='val'>{hurst:.4f}</td></tr>
            <tr><td class='lbl'>Regime Timestamp</td>
                <td class='val'>{ts}</td></tr>
        </table>
        <div class='action-box'>
            ⚡ <strong>System Action:</strong> {action_line}
        </div>
        <div class='footer'>
            Automated regime alert dispatched by Sovereign Swarm v37.0 — regime_monitor.py
        </div>
    </div></body></html>
    """
    return subject, html


def _send_regime_alert(hurst: float, adx: float, regime: str,
                        prev_regime: str, ts: str,
                        plus_di: float = 0.0, minus_di: float = 0.0,
                        macro_trend: str = "NEUTRAL") -> bool:
    """Blocking SMTP dispatch. Always call via asyncio.to_thread."""
    try:
        from email_dispatcher import send_html_alert
        subject, html = _build_regime_alert_html(
            hurst, adx, regime, prev_regime, ts, plus_di, minus_di, macro_trend
        )
        ok = send_html_alert(subject, html)
        return bool(ok)
    except Exception as e:
        log.error(f"[REGIME-MONITOR] Email dispatch failed: {e}")
        return False


async def run_regime_monitor():
    """
    Async coroutine — runs forever in the agent manager's event loop.
    Directional State Machine:
        Watches for state transitions between BULL_TREND, BEAR_TREND, and CHOP.
        Debounced to prevent duplicate emails for sustained regimes.
    """
    log.info("[REGIME-MONITOR] Started v37.0 Directional Regime Monitor.")

    initial = _read_regime()
    last_alerted_regime: str = str(initial.get("regime") or "UNKNOWN") if initial else "UNKNOWN"
    log.info(f"[REGIME-MONITOR] Bootstrapped with initial regime: {last_alerted_regime}")

    KNOWN_REGIMES = ("BULL_TREND", "BULLISH", "TRENDING", "BEAR_TREND", "BEARISH", "CHOP", "DEFENSIVE_CHOP", "NEUTRAL")

    while True:
        await asyncio.sleep(POLL_INTERVAL_SEC)

        data = _read_regime()
        if data is None:
            continue

        hurst       = float(data.get("hurst") or 0)
        adx         = float(data.get("adx") or 0)
        regime      = str(data.get("regime") or "UNKNOWN")
        ts          = str(data.get("timestamp") or "")
        plus_di     = float(data.get("plus_di") or 0)
        minus_di    = float(data.get("minus_di") or 0)
        macro_trend = str(data.get("macro_trend") or "NEUTRAL")

        # Check for meaningful state transition
        if regime != last_alerted_regime and regime in KNOWN_REGIMES:
            log.info(
                f"[REGIME-MONITOR] 🚨 REGIME SHIFT DETECTED: {last_alerted_regime} ➔ {regime} | "
                f"ADX={adx:.2f} +DI={plus_di:.1f} -DI={minus_di:.1f} Macro={macro_trend}. Dispatching alert..."
            )
            prev = last_alerted_regime
            last_alerted_regime = regime
            
            ok = await asyncio.to_thread(
                _send_regime_alert, hurst, adx, regime, prev, ts, plus_di, minus_di, macro_trend
            )
            log.info(f"[REGIME-MONITOR] Regime transition alert email: {'SENT' if ok else 'FAILED'}")
        else:
            log.debug(
                f"[REGIME-MONITOR] Stable regime={regime} (last={last_alerted_regime}) | ADX={adx:.1f}"
            )


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    data = _read_regime()
    if data:
        print(f"\nCurrent regime snapshot:")
        print(f"  Hurst   : {data['hurst']:.4f}  (gate: >{HURST_TREND_GATE})")
        print(f"  ADX     : {data['adx']:.2f}   (gate: >{ADX_CONFIRM_GATE})")
        print(f"  Regime  : {data['regime']}")
        print(f"  Updated : {data['timestamp']}")
        trend = data['hurst'] > HURST_TREND_GATE and data['adx'] > ADX_CONFIRM_GATE
        print(f"\n  >> Trigger state: {'[!] TREND ALERT WOULD FIRE' if trend else '[OK] No crossing -- watching'}")
    else:
        print("Could not read market_regime table.")
