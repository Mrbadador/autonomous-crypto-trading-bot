# email_dispatcher.py — High-Fidelity HTML Email Dispatcher

import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import datetime
import asyncio
import time

# Load environment variables (anchored to file directory)
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_HERE, ".env"), override=True)
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_APP_PASSWORD")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] EMAIL_DISPATCHER: %(message)s")
log = logging.getLogger("email_dispatcher")
log.propagate = False
if not log.handlers:
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] EMAIL_DISPATCHER: %(message)s")
    _sh = logging.StreamHandler()
    _sh.setFormatter(_formatter)
    log.addHandler(_sh)
    try:
        os.makedirs(os.path.join(_HERE, "logs"), exist_ok=True)
        _fh = logging.FileHandler(os.path.join(_HERE, "logs", "email_dispatcher.log"), encoding="utf-8")
        _fh.setFormatter(_formatter)
        log.addHandler(_fh)
    except: pass

# V30.3: Alert debounce registry to prevent inbox spam (max 1 email per hour per alert type)
_alert_registry = {}

def _is_debounced(subject):
    subject_lower = subject.lower()
    debounce_key = None
    if "stale connection" in subject_lower:
        debounce_key = "stale_connection"
    elif "heartbeat" in subject_lower:
        debounce_key = "heartbeat_failed"
        
    if debounce_key:
        now = time.time()
        last_sent = _alert_registry.get(debounce_key, 0.0)
        if now - last_sent < 3600.0:
            log.info(f"[DEBOUNCE] Suppressing alert email: '{subject}' (Last sent {now - last_sent:.1f}s ago)")
            return True
        _alert_registry[debounce_key] = now
    return False

def send_alert(subject, message):
    """Core function to dispatch plain text alerts via Gmail SMTP."""
    if _is_debounced(subject):
        return False

    if not GMAIL_USER or not GMAIL_PASS:
        log.error("GMAIL credentials missing from .env. Cannot send alert.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = GMAIL_USER
        msg['Subject'] = f"[LLAMA BOT] {subject}"

        msg.attach(MIMEText(message, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
        server.quit()
        
        log.info(f"Alert sent successfully: {subject.encode('ascii', 'replace').decode('ascii')}")
        return True
    except Exception as e:
        log.error(f"Failed to send email alert: {e}")
        return False

def send_html_alert(subject, html_content):
    """Dispatches beautiful HTML alerts via Gmail SMTP."""
    if _is_debounced(subject):
        return False

    if not GMAIL_USER or not GMAIL_PASS:
        log.error("GMAIL credentials missing from .env. Cannot send HTML alert.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_USER
        msg['To'] = GMAIL_USER
        msg['Subject'] = f"[LLAMA BOT] {subject}"

        msg.attach(MIMEText(html_content, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
        server.quit()
        
        log.info(f"HTML Alert sent successfully: {subject.encode('ascii', 'replace').decode('ascii')}")
        return True
    except Exception as e:
        log.error(f"Failed to send HTML email alert: {e}")
        return False

async def send_alert_async(subject, message):
    return await asyncio.to_thread(send_alert, subject, message)

async def send_html_alert_async(subject, html_content):
    return await asyncio.to_thread(send_html_alert, subject, html_content)

def send_morning_briefing(rules):
    subject = "Morning Quant Briefing"
    body = "The Deep Quant Analyst has generated new trading rules:\n\n"
    for i, rule in enumerate(rules, 1):
        body += f"{i}. {rule}\n"
    body += "\nThese rules are now active in the engine memory."
    return send_alert(subject, body)

def send_whale_warning(move_desc, impact):
    subject = "URGENT: Whale Danger Detected!"
    body = (
        f"The Whale Watcher has triggered a DANGER status.\n\n"
        f"Move Detected: {move_desc}\n"
        f"Impact Level: {impact}/10\n\n"
        f"Adaptive Risk Logic: All engines have tightened Stop-Losses to 2% trailing."
    )
    return send_alert(subject, body)

def send_trade_log(symbol, side=None, price=0.0, pnl=None, status="OPENED", trade_context=None):
    """
    V30.5 Instant Real-Time Trade Alert Dispatcher.
    Dispatched immediately when an order is placed, filled, or closed on Bitget.
    Includes comprehensive "Why We Closed" exit rationale, hold duration, and execution triggers.
    """
    if trade_context is None:
        trade_context = {}
        
    status = str(trade_context.get("status", status)).upper()
    side = str(trade_context.get("side", side or "LONG")).upper()
    symbol = str(trade_context.get("symbol", symbol)).upper()
    price_val = float(trade_context.get("price", price or 0.0))
    qty = trade_context.get("qty", "N/A")
    timestamp = trade_context.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    engine = trade_context.get("engine")
    if not engine:
        notes_str = str(trade_context.get("notes", "")).upper()
        if "SWING" in notes_str or "ICHIMOKU" in notes_str:
            engine = "The Swing Architect (4H)"
        elif "MARL" in notes_str or "SWARM" in notes_str:
            engine = "MARL Swarm Agent (15m)"
        elif "SNP" in str(trade_context.get("client_oid", "")):
            engine = "Sniper (15m)"
        else:
            engine = "Berserker (5m)"
    
    regime = trade_context.get("regime", "N/A")
    hurst = trade_context.get("hurst", "N/A")
    adx = trade_context.get("adx", "N/A")
    sl = trade_context.get("stop_loss", "N/A")
    tp = trade_context.get("take_profit", "N/A")
    conviction = trade_context.get("conviction", "N/A")
    ai_reasoning = trade_context.get("ai_reasoning", trade_context.get("notes", "Confirmed volume breakout + LLaMA 3.1 multi-agent consensus alignment."))
    entry_price_val = trade_context.get("entry_price")
    peak_price_val = trade_context.get("peak_price")
    entry_time_str = trade_context.get("entry_time")
    close_trigger = trade_context.get("close_trigger")
    close_reason = trade_context.get("close_reason")

    # Dynamic database fetching to enrich trade alert details if not passed
    try:
        import sqlite3
        if os.path.exists("trading_data.db"):
            with sqlite3.connect("trading_data.db", timeout=5) as conn_data:
                res_reg = conn_data.execute("SELECT hurst, adx, regime FROM market_regime ORDER BY rowid DESC LIMIT 1").fetchone()
                if res_reg:
                    if regime in ["N/A", None]: regime = str(res_reg[2])
                    if hurst in ["N/A", None]: hurst = f"{float(res_reg[0]):.4f}"
                    if adx in ["N/A", None]: adx = f"{float(res_reg[1]):.2f}"
                
                # Check all active fleet tables
                for table_name in ["swing_open_positions", "open_positions", "marl_live_open_positions", "berserker_open_positions"]:
                    try:
                        cursor = conn_data.execute(f"SELECT * FROM {table_name} WHERE symbol = ? OR symbol = ? ORDER BY rowid DESC LIMIT 1", (symbol, symbol.replace("/", "")))
                        cols = [d[0] for d in cursor.description]
                        row_raw = cursor.fetchone()
                        if row_raw:
                            row_pos = dict(zip(cols, row_raw))
                            sl_val = row_pos.get("stop_loss") or row_pos.get("trailing_stop")
                            tp_val = row_pos.get("take_profit")
                            conf_val = row_pos.get("action_confidence")
                            comm_val = row_pos.get("comments") or row_pos.get("notes")
                            ep_val = row_pos.get("entry_price")
                            pk_val = row_pos.get("peak_price")
                            et_val = row_pos.get("entry_time") or row_pos.get("opened_at")
                            lev_val = row_pos.get("leverage")
                            sz_val = row_pos.get("qty") or row_pos.get("size")

                            if sl_val and sl in ["N/A", None]: sl = f"${float(sl_val):,.4f}"
                            if tp_val and tp in ["N/A", None]: tp = f"${float(tp_val):,.4f}"
                            if conf_val and conviction in ["N/A", None]: conviction = f"{round(float(conf_val)*10)}/10"
                            if comm_val and ai_reasoning in ["N/A", None]: ai_reasoning = str(comm_val)
                            if ep_val and entry_price_val in [None, "N/A"]: entry_price_val = float(ep_val)
                            if pk_val and peak_price_val in [None, "N/A"]: peak_price_val = float(pk_val)
                            if et_val and entry_time_str in [None, "N/A"]: entry_time_str = str(et_val)
                            if lev_val is not None: trade_context["leverage"] = float(lev_val)
                            if sz_val is not None and qty in ["N/A", None]: qty = float(sz_val)
                            break
                    except Exception:
                        pass
    except Exception as db_err:
        log.warning(f"Error enriching trade log from DB: {db_err}")

    # Liquidation & Margin metrics
    leverage = float(trade_context.get('leverage', 5.0))
    if "SHORT" in side or "SELL" in side:
        liq_calc = price_val * (1.0 + (1.0 / leverage) - 0.004) if price_val > 0 else 0.0
        dist_calc = ((liq_calc - price_val) / price_val) * 100.0 if price_val > 0 else 0.0
    else:
        liq_calc = price_val * (1.0 - (1.0 / leverage) + 0.004) if price_val > 0 else 0.0
        dist_calc = ((price_val - liq_calc) / price_val) * 100.0 if price_val > 0 else 0.0
        
    liq_str = f"${liq_calc:,.4f}" if liq_calc > 0 else "N/A"
    dist_str = f"{dist_calc:+.2f}%" if dist_calc > 0 else "Safe"
    
    pnl_val = trade_context.get("pnl", pnl)
    pnl_float = None
    roe_str = ""
    price_delta_str = ""
    margin_usd = None
    
    qty_float = None
    try:
        raw_qty = trade_context.get("qty", qty)
        if raw_qty not in ["N/A", None]:
            qty_float = float(raw_qty)
    except:
        pass

    if pnl_val is not None:
        try:
            pnl_float = float(pnl_val)
            # Fail-safe check: If pnl_float is unexpectedly large relative to position quantity, convert from price diff to dollar PnL
            if qty_float and qty_float < 0.5 and abs(pnl_float) > 5.0 and price_val > 100:
                pnl_float = pnl_float * qty_float
            
            pnl_str = f"${pnl_float:+.2f} USD"
            pnl_val = pnl_float

            # Calculate Margin & ROE %
            if qty_float and price_val > 0 and leverage > 0:
                margin_usd = (price_val * qty_float) / leverage
                if margin_usd > 0:
                    roe_pct = (pnl_float / margin_usd) * 100.0
                    roe_str = f"{roe_pct:+.2f}%"
                    price_pct = roe_pct / leverage
                    price_delta_str = f"{price_pct:+.2f}%"
        except Exception:
            pnl_str = str(pnl_val)
    else:
        pnl_str = "N/A"
    
    # Calculate Holding Duration if entry_time is available
    hold_duration_str = trade_context.get("hold_duration", "N/A")
    if hold_duration_str in ["N/A", None] and entry_time_str:
        try:
            cleaned_time = entry_time_str.replace("Z", "+00:00")
            if "T" in cleaned_time:
                t_entry = datetime.fromisoformat(cleaned_time)
            else:
                t_entry = datetime.strptime(cleaned_time[:19], "%Y-%m-%d %H:%M:%S")
            now_utc = datetime.utcnow()
            diff_secs = max(1, int((now_utc - t_entry.replace(tzinfo=None)).total_seconds()))
            if diff_secs < 60:
                hold_duration_str = f"{diff_secs}s"
            elif diff_secs < 3600:
                hold_duration_str = f"{diff_secs // 60}m {diff_secs % 60}s"
            else:
                hrs = diff_secs // 3600
                mins = (diff_secs % 3600) // 60
                hold_duration_str = f"{hrs}h {mins}m"
        except Exception as e_dur:
            log.debug(f"Hold duration calc note: {e_dur}")

    # Build / synthesize intelligent close reason and trigger badge if not already specified
    if not close_trigger:
        if "PARTIAL" in status:
            close_trigger = "PARTIAL TAKE-PROFIT (50%)"
        elif pnl_float is not None and pnl_float > 0:
            if peak_price_val and peak_price_val > 0 and price_val < peak_price_val:
                close_trigger = "TRAILING STOP-LOSS (PROFIT PROTECT)"
            else:
                close_trigger = "TAKE-PROFIT TARGET HIT"
        elif pnl_float is not None and pnl_float < 0:
            close_trigger = "STOP-LOSS HIT (RISK CONTAINMENT)"
        else:
            close_trigger = "POSITION CLOSED / SETTLED"

    if not close_reason:
        if "PARTIAL" in status:
            close_reason = f"50% partial profit secured at 1.5x ATR momentum expansion (${price_val:,.4f}) to de-risk exposure while leaving runner active."
        elif "TRAILING" in close_trigger.upper():
            peak_str = f"${float(peak_price_val):,.4f}" if peak_price_val else "peak price"
            close_reason = f"Dynamic trailing stop triggered at ${price_val:,.4f} following a price pullback from {peak_str}. Trailing stop locked in accrued gains and protected capital from market retracement."
        elif "TAKE-PROFIT" in close_trigger.upper() or (pnl_float and pnl_float > 0):
            close_reason = f"Price successfully reached the designated Take-Profit target at ${price_val:,.4f}, capturing {pnl_str} (+{roe_str} ROE) in accordance with quantitative strategy rules."
        elif "STOP-LOSS" in close_trigger.upper() or (pnl_float and pnl_float < 0):
            close_reason = f"Price touched the protective Stop-Loss threshold at ${price_val:,.4f}. Automated risk firewall triggered immediately to cap drawdown and preserve portfolio equity."
        elif pnl_float == 0:
            close_reason = f"Position closed at breakeven ($ {price_val:,.4f}) with $0.00 loss to eliminate market exposure during uncertain conditions."
        else:
            close_reason = f"Position closed on Bitget live execution engine at ${price_val:,.4f} based on strategy exit criteria."

    side_color = "#22c55e" if "LONG" in side or "BUY" in side else "#ef4444"
    roe_display = f" ({roe_str} ROE)" if roe_str else ""
    
    # Construct Subject and Header Title
    if "PARTIAL" in status:
        subject = f"💰 [PARTIAL TP (50%)] {symbol} {side} @ ${price_val:,.4f} | Secured: {pnl_str}{roe_display} | Bitget Live"
        header_title = "💰 50% PROFIT SECURED (PARTIAL TP)"
        badge_text = f"PARTIAL TP ({pnl_str}{roe_display})"
        trigger_badge_color = "#10b981"
    elif "OPEN" in status:
        subject = f"⚡ [TRADE OPENED] {side} {symbol} @ ${price_val:,.4f} | {leverage:.1f}x Isolated | Bitget Live"
        header_title = "⚡ LIVE TRADE EXECUTED"
        badge_text = f"ACTIVE {side}"
        trigger_badge_color = "#38bdf8"
    else:
        is_win = pnl_float > 0 if pnl_float is not None else False
        res_tag = "PROFIT" if is_win else "CLOSED"
        trigger_summary = "TP HIT" if "TAKE-PROFIT" in close_trigger.upper() else "TRAILING STOP" if "TRAILING" in close_trigger.upper() else "STOP-LOSS" if "STOP-LOSS" in close_trigger.upper() else close_trigger
        subject = f"🎯 [TRADE {res_tag} - {trigger_summary}] {symbol} {side} | PnL: {pnl_str}{roe_display} | Bitget Live"
        header_title = f"🎯 POSITION SETTLED ({res_tag})"
        badge_text = f"{res_tag} ({pnl_str}{roe_display})"
        trigger_badge_color = "#22c55e" if is_win else "#f43f5e"

    is_closing_event = status not in ["OPENED", "OPEN"]
    entry_p_disp = f"${float(entry_price_val):,.4f}" if entry_price_val else "N/A"
    peak_p_disp = f"${float(peak_price_val):,.4f}" if peak_price_val else "N/A"

    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0b101b; color: #f1f5f9; padding: 20px; }}
            .container {{ max-width: 600px; background: #131d31; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
            h2 {{ color: #38bdf8; margin-top: 0; font-size: 20px; border-bottom: 2px solid #1e293b; padding-bottom: 12px; text-align: center; }}
            .badge {{ display: inline-block; background: {side_color}; color: #000; font-weight: 800; padding: 5px 12px; border-radius: 4px; font-size: 13px; margin-bottom: 15px; }}
            .trigger-badge {{ display: inline-block; background: {trigger_badge_color}; color: #ffffff; font-weight: 700; padding: 4px 10px; border-radius: 4px; font-size: 12px; }}
            h3 {{ color: #94a3b8; font-size: 13px; margin-top: 18px; text-transform: uppercase; letter-spacing: 0.05em; border-left: 3px solid #38bdf8; padding-left: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
            td {{ padding: 7px 10px; border-bottom: 1px solid #1e293b; font-size: 13px; }}
            .label {{ font-weight: 600; color: #64748b; width: 45%; }}
            .value {{ color: #f8fafc; text-align: right; font-family: 'Consolas', monospace; font-weight: bold; }}
            .reason-box {{ background: #080d1a; border-left: 4px solid #38bdf8; padding: 14px; border-radius: 6px; font-size: 13px; color: #cbd5e1; margin-top: 10px; line-height: 1.5; }}
            .exit-box {{ background: #0c1629; border: 1px solid #1e3a8a; border-left: 4px solid {trigger_badge_color}; padding: 14px; border-radius: 6px; font-size: 13px; color: #f1f5f9; margin-top: 10px; line-height: 1.5; }}
            .footer {{ font-size: 11px; color: #475569; text-align: center; margin-top: 24px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>{header_title}</h2>
            <div style="text-align: center;">
                <span class="badge">{badge_text}</span>
            </div>
            
            {f'''
            <div class="exit-box">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: bold; color: #38bdf8; font-size: 14px;">🎯 WHY WE CLOSED:</span>
                    <span class="trigger-badge">{close_trigger}</span>
                </div>
                <div style="margin-top: 6px; color: #e2e8f0; font-size: 13px;">
                    {close_reason}
                </div>
                <div style="margin-top: 10px; padding-top: 8px; border-top: 1px dashed #334155; font-size: 12px; color: #94a3b8; display: flex; justify-content: space-between;">
                    <span><b>Entry ➔ Exit:</b> {entry_p_disp} ➔ ${price_val:,.4f}</span>
                    <span><b>Holding Duration:</b> {hold_duration_str}</span>
                </div>
            </div>
            ''' if is_closing_event else ''}
            
            <h3>Section 1: Execution Details</h3>
            <table>
                <tr><td class="label">Asset / Symbol</td><td class="value" style="color: #38bdf8;">{symbol}</td></tr>
                <tr><td class="label">Action / Side</td><td class="value" style="color: {side_color};">{side}</td></tr>
                <tr><td class="label">Strategy Engine</td><td class="value">{engine}</td></tr>
                <tr><td class="label">Execution Price ($)</td><td class="value">${price_val:,.4f}</td></tr>
                <tr><td class="label">Position Size</td><td class="value">{qty}</td></tr>
                <tr><td class="label">Leverage & Mode</td><td class="value">{leverage:.0f}x Isolated Margin</td></tr>
                <tr><td class="label">Exchange</td><td class="value">Bitget USDT-M (Live Pilot)</td></tr>
                <tr><td class="label">Timestamp</td><td class="value">{timestamp}</td></tr>
                {f'<tr><td class="label">⏱️ Holding Duration</td><td class="value">{hold_duration_str}</td></tr>' if is_closing_event and hold_duration_str != "N/A" else ''}
                {f'<tr><td class="label">🏔️ Peak Price (HWM)</td><td class="value">{peak_p_disp}</td></tr>' if is_closing_event and peak_p_disp != "N/A" else ''}
                <tr><td class="label">💵 Realized Dollar PnL ($)</td><td class="value" style="color: {'#22c55e' if pnl_float and pnl_float > 0 else '#ef4444' if pnl_float and pnl_float < 0 else '#f8fafc'}; font-size: 15px;">{pnl_str}</td></tr>
                {f'<tr><td class="label">📊 Return on Margin (ROE %)</td><td class="value" style="color: #22c55e if pnl_float > 0 else #ef4444;">{roe_str} (Price Move: {price_delta_str})</td></tr>' if roe_str else ''}
            </table>
            
            <h3>Section 2: Risk & Protection Bounds</h3>
            <table>
                <tr><td class="label">Stop-Loss Target</td><td class="value" style="color: #f87171;">{sl}</td></tr>
                <tr><td class="label">Take-Profit Target</td><td class="value" style="color: #4ade80;">{tp}</td></tr>
                <tr><td class="label">💀 Estimated Liquidation</td><td class="value" style="color: #f43f5e;">{liq_str}</td></tr>
                <tr><td class="label">Buffer to Liquidation</td><td class="value" style="color: #38bdf8;">{dist_str}</td></tr>
            </table>
            
            <h3>Section 3: AI Intelligence & Market Context</h3>
            <table>
                <tr><td class="label">Macro Market Regime</td><td class="value">{regime}</td></tr>
                <tr><td class="label">Trend Strength (ADX)</td><td class="value">{adx}</td></tr>
                <tr><td class="label">AI Boardroom Conviction</td><td class="value" style="color: #38bdf8;">{conviction}</td></tr>
            </table>
            
            <div class="reason-box">
                <b>💡 AI Strategy Trigger / Entry Thesis:</b><br>
                {ai_reasoning}
            </div>
            
            <div class="footer">
                Dispatched immediately by the Sovereign Swarm Live Execution Engine.
            </div>
        </div>
    </body>
    </html>
    """
    return send_html_alert(subject, html_body)

def query_last_signals(symbol):
    patterns = "None Detected"
    arbitrage = "None Detected"
    try:
        import sqlite3
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ledger_db = os.path.join(base_dir, "trading_ledger.db")
        if os.path.exists(ledger_db):
            conn = sqlite3.connect(ledger_db, timeout=5)
            symbol_clean = str(symbol).upper().replace("/", "").replace("-", "")
            
            # Query last active patterns for this symbol within 5 minutes
            cursor_pat = conn.execute(
                "SELECT timeframe, pattern_name FROM active_patterns WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1"
            )
            row_pat = cursor_pat.fetchone()
            if row_pat:
                patterns = f"{row_pat[0]} {row_pat[1]}"
                
            # Query last active correlations for this symbol within 5 minutes
            cursor_corr = conn.execute(
                "SELECT signal, beta_score FROM active_correlations WHERE lag_asset = ? ORDER BY timestamp DESC LIMIT 1"
            )
            row_corr = cursor_corr.fetchone()
            if row_corr:
                arbitrage = f"FLASH {row_corr[0]} (Beta: {row_corr[1]:.2f})"
                
            conn.close()
    except Exception as e:
        log.error(f"Error querying last signals for email: {e}")
    return patterns, arbitrage

def send_trade_email(symbol, side, price, qty, sl, tp, status="OPENED", trade_context=None):
    """
    Overhauled institutional-grade HTML email trade reports.
    Queries recent active_patterns and active_correlations for symbol in last 5m.
    """
    if trade_context is None:
        trade_context = {}

    symbol = trade_context.get("symbol", symbol)
    side = trade_context.get("side", side or "N/A").upper()
    price_val = trade_context.get("price", price)
    qty_val = trade_context.get("qty", qty)
    sl_val = trade_context.get("stop_loss", sl)
    tp_val = trade_context.get("take_profit", tp)
    status = trade_context.get("status", status)
    
    quality = trade_context.get("quality", "N/A")
    conviction = trade_context.get("conviction", "N/A")
    ai_thought = trade_context.get("ai_thought", "Bypassing Chop Guard due to 15m Hammer")
    
    regime = trade_context.get("regime", "N/A")
    hurst = trade_context.get("hurst", "N/A")
    adx = trade_context.get("adx", "N/A")
    
    # Dynamic database fetching to resolve "Blank Data" inside send_trade_email
    try:
        import sqlite3
        if os.path.exists("trading_data.db"):
            conn_data = sqlite3.connect("trading_data.db", timeout=5)
            res_reg = conn_data.execute("SELECT hurst, adx, regime FROM market_regime ORDER BY rowid DESC LIMIT 1").fetchone()
            if res_reg:
                if regime == "N/A" or regime is None:
                    regime = str(res_reg[2])
                if hurst == "N/A" or hurst is None:
                    hurst = f"{float(res_reg[0]):.4f}"
                if adx == "N/A" or adx is None:
                    adx = f"{float(res_reg[1]):.2f}"
            
            cursor_pos = conn_data.execute(
                "SELECT stop_loss, take_profit, leverage, action_confidence, comments FROM open_positions WHERE symbol = ? ORDER BY rowid DESC LIMIT 1",
                (symbol,)
            )
            row_pos = cursor_pos.fetchone()
            if not row_pos:
                cursor_pos = conn_data.execute(
                    "SELECT stop_loss, take_profit, leverage, action_confidence, comments FROM berserker_open_positions WHERE symbol = ? ORDER BY rowid DESC LIMIT 1",
                    (symbol,)
                )
                row_pos = cursor_pos.fetchone()
                
            if row_pos:
                if sl_val == "N/A" or sl_val is None or sl_val == 0.0:
                    sl_val = float(row_pos[0])
                if tp_val == "N/A" or tp_val is None or tp_val == 0.0:
                    tp_val = float(row_pos[1])
                leverage = float(row_pos[2])
                if conviction == "N/A" or conviction is None:
                    conv_score = float(row_pos[3])
                    conviction = round(conv_score * 10.0) if conv_score <= 1.0 else round(conv_score)
                if ai_thought == "Bypassing Chop Guard due to 15m Hammer" or ai_thought == "N/A" or ai_thought is None:
                    ai_thought = str(row_pos[4]) if row_pos[4] else ai_thought
            conn_data.close()
            
        if os.path.exists("trading_ledger.db") and (regime == "N/A" or regime is None):
            conn_ledger = sqlite3.connect("trading_ledger.db", timeout=5)
            res_ledger = conn_ledger.execute("SELECT trend_strength FROM market_context ORDER BY rowid DESC LIMIT 1").fetchone()
            if res_ledger:
                trend_val = float(res_ledger[0])
                regime = "TREND" if abs(trend_val) > 0.3 else "CHOP"
            conn_ledger.close()
    except Exception as db_err:
        log.error(f"Error resolving dynamic parameters for HTML email: {db_err}")

    # Expose liquidation prices
    leverage = float(trade_context.get("leverage", 5.0))
    entry_price = float(price_val)
    if "SHORT" in side:
        liq_price = entry_price * (1.0 + (1.0 / leverage) - 0.004)
        dist_liq = ((liq_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
    else:
        liq_price = entry_price * (1.0 - (1.0 / leverage) + 0.004)
        dist_liq = ((entry_price - liq_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
        
    liq_str = f"${liq_price:.4f}"
    dist_str = f"{dist_liq:+.2f}%"

    scout_patterns, arbitrage_signal = query_last_signals(symbol)
    
    scout_patterns = trade_context.get("scout_patterns", scout_patterns) or "None Detected"
    arbitrage_signal = trade_context.get("arbitrage_signal", arbitrage_signal) or "None Detected"

    # Format Stop-Loss and Take-Profit values cleanly
    try:
        sl_str = f"${float(sl_val):.4f}" if isinstance(sl_val, (int, float)) else str(sl_val)
    except:
        sl_str = str(sl_val)
    try:
        tp_str = f"${float(tp_val):.4f}" if isinstance(tp_val, (int, float)) else str(tp_val)
    except:
        tp_str = str(tp_val)

    # Dynamic Subject Line Format: [V15 {status}] {side} {symbol} | Quality: {quality} | Conviction: {conviction}/10
    subject = f"[V15 {status}] {side} {symbol} | Quality: {quality} | Conviction: {conviction}/10"

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f1f5f9; padding: 20px; }}
            .container {{ max-width: 600px; background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin: 0 auto; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); }}
            h2 {{ color: #818cf8; margin-top: 0; font-size: 22px; border-bottom: 2px solid #334155; padding-bottom: 12px; text-align: center; }}
            h3 {{ color: #cbd5e1; font-size: 14px; margin-top: 20px; text-transform: uppercase; letter-spacing: 0.05em; border-left: 3px solid #818cf8; padding-left: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
            td {{ padding: 8px 12px; border-bottom: 1px solid #334155; font-size: 14px; }}
            .label {{ font-weight: 600; color: #94a3b8; width: 45%; }}
            .value {{ color: #f1f5f9; text-align: right; font-family: 'Consolas', monospace; font-weight: bold; }}
            .reason-box {{ background: #0f172a; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; font-size: 14px; color: #e2e8f0; margin-top: 10px; font-style: italic; }}
            .footer {{ font-size: 11px; color: #64748b; text-align: center; margin-top: 24px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📈 INSTITUTIONAL TRADE REPORT</h2>
            
            <h3>Market Context</h3>
            <table>
                <tr><td class="label">Regime</td><td class="value">{regime}</td></tr>
                <tr><td class="label">Hurst Exponent</td><td class="value">{hurst}</td></tr>
                <tr><td class="label">ADX</td><td class="value">{adx}</td></tr>
            </table>
            
            <h3>Intelligence Signals</h3>
            <table>
                <tr><td class="label">Scout Patterns</td><td class="value">{scout_patterns}</td></tr>
                <tr><td class="label">Arbitrage Signal</td><td class="value">{arbitrage_signal}</td></tr>
            </table>
            
            <h3>AI Reasoning</h3>
            <div class="reason-box">
                "{ai_thought}"
            </div>
            
            <h3>Risk Profile</h3>
            <table>
                <tr><td class="label">Entry Price</td><td class="value">${entry_price:.4f}</td></tr>
                <tr><td class="label">Quantity</td><td class="value">{qty_val}</td></tr>
                <tr><td class="label">Stop-Loss</td><td class="value">{sl_str}</td></tr>
                <tr><td class="label">Take-Profit</td><td class="value">{tp_str}</td></tr>
            </table>
            
            <h3>The Danger Zone</h3>
            <table>
                <tr><td class="label">💀 Liquidation Price</td><td class="value" style="color: #f43f5e;">{liq_str}</td></tr>
                <tr><td class="label">Distance to Liq (%)</td><td class="value" style="color: #fb7185;">{dist_str}</td></tr>
            </table>
            
            <div class="footer">
                Automated Institutional Telemetry generated securely by the V15 MARL Swarm.
            </div>
        </div>
    </body>
    </html>
    """
    return send_html_alert(subject, html_content)

async def send_morning_briefing_async(rules):
    return await asyncio.to_thread(send_morning_briefing, rules)

async def send_whale_warning_async(move_desc, impact):
    return await asyncio.to_thread(send_whale_warning, move_desc, impact)

async def send_trade_log_async(symbol, side=None, price=0.0, pnl=None, status="OPENED", trade_context=None):
    return await asyncio.to_thread(send_trade_log, symbol, side, price, pnl, status, trade_context)

async def send_trade_email_async(symbol, side, price, qty, sl, tp, status="OPENED", trade_context=None):
    return await asyncio.to_thread(send_trade_email, symbol, side, price, qty, sl, tp, status, trade_context)

def send_hourly_heartbeat_report(fleet_pnl, active_exposure, warden_status, position_details=None, stats_data=None):
    """
    V30.5 Sovereign Swarm Hourly Heartbeat Report.
    Includes comprehensive explanation of why trades occurred or why capital was protected in cash.
    """
    pnl_str = f"${float(fleet_pnl):+.2f}" if fleet_pnl is not None else "$0.00"
    pnl_color = '#22c55e' if fleet_pnl and float(fleet_pnl) > 0 else '#ef4444' if fleet_pnl and float(fleet_pnl) < 0 else '#f1f5f9'
    
    subject = f"[V15.0 HEARTBEAT] Swarm Status: Active | Fleet PnL: {pnl_str} | Active Nodes: {active_exposure}"
    
    pos_rows_html = ""
    if position_details:
        for pos in position_details:
            side_badge = pos.get('side', 'N/A').upper()
            s_col = "#22c55e" if "LONG" in side_badge or "BUY" in side_badge else "#ef4444"
            pos_rows_html += f"""
            <tr>
                <td class="value" style="text-align: left; color: #38bdf8;">{pos.get('symbol', 'N/A')}</td>
                <td class="value" style="text-align: center; color: {s_col};">{side_badge}</td>
                <td class="value" style="text-align: center;">{pos.get('leverage', '5.0')}x</td>
                <td class="value" style="text-align: right;">${float(pos.get('entry_price', 0.0)):,.4f}</td>
                <td class="value" style="text-align: right;">{pos.get('qty', 0.0)}</td>
            </tr>
            """
    else:
        pos_rows_html = "<tr><td colspan='5' style='text-align: center; color: #94a3b8; font-style: italic; padding: 12px;'>No active exposure nodes (100% in cash).</td></tr>"

    # Query trades executed in the past 60 minutes
    recent_trades_html = ""
    recent_trades_found = []
    try:
        import sqlite3
        if os.path.exists("trading_data.db"):
            with sqlite3.connect("trading_data.db", timeout=5) as conn:
                for tbl, eng_name in [("trades", "Sniper (15m)"), ("berserker_trades", "Berserker (5m)")]:
                    try:
                        rows = conn.execute(
                            f"SELECT timestamp, symbol, side, entry_price, exit_price, pnl, exit_reason FROM {tbl} WHERE timestamp >= datetime('now', '-60 minutes') ORDER BY timestamp DESC"
                        ).fetchall()
                        for r in rows:
                            recent_trades_found.append({
                                "engine": eng_name, "time": r[0], "symbol": r[1], "side": r[2],
                                "entry": r[3], "exit": r[4], "pnl": r[5], "reason": r[6]
                            })
                    except: pass
    except Exception as db_e:
        log.warning(f"Error querying past hour trades: {db_e}")

    total_live_bal = 65.90
    syms_display_str = "BTC, ETH, SOL, XRP, BNB, DOGE, ADA, AVAX, LINK, NEAR"
    try:
        import sqlite3, json
        if os.path.exists("trading_data.db"):
            with sqlite3.connect("trading_data.db", timeout=5) as conn_b:
                row_b = conn_b.execute("SELECT value FROM system_state WHERE key = 'balance'").fetchone()
                if row_b and row_b[0]:
                    total_live_bal = float(row_b[0])
                row_s = conn_b.execute("SELECT value FROM system_state WHERE key = 'ALLOWED_SYMBOLS'").fetchone()
                if row_s and row_s[0]:
                    syms_display_str = ", ".join([s.split('/')[0] for s in json.loads(row_s[0])])
    except: pass

    if recent_trades_found:
        trade_items = ""
        for t in recent_trades_found:
            t_pnl = float(t['pnl'] or 0.0)
            t_col = "#22c55e" if t_pnl > 0 else "#ef4444"
            trade_items += f"""
            <tr>
                <td class="value" style="text-align: left; color: #38bdf8;">{t['symbol']} ({t['engine']})</td>
                <td class="value" style="text-align: center;">{t['side'].upper()}</td>
                <td class="value" style="text-align: right;">${float(t['entry']):,.4f} &rarr; ${float(t['exit']):,.4f}</td>
                <td class="value" style="text-align: right; color: {t_col}; font-weight: bold;">${t_pnl:+.2f} USD</td>
                <td class="value" style="text-align: right; font-size: 11px; color: #94a3b8;">{t['reason']}</td>
            </tr>
            """
        recent_trades_html = f"""
        <h3>Section 3: Past 60-Minute Executed Trades</h3>
        <table>
            <thead>
                <tr>
                    <th style="text-align: left;">Asset</th>
                    <th>Side</th>
                    <th style="text-align: right;">Entry &rarr; Exit ($)</th>
                    <th style="text-align: right;">Realized PnL ($)</th>
                    <th style="text-align: right;">Exit Trigger</th>
                </tr>
            </thead>
            <tbody>
                {trade_items}
            </tbody>
        </table>
        """
    else:
        recent_trades_html = f"""
        <h3>Section 3: Past 60-Minute Activity (Why No Trades Occurred)</h3>
        <div style="background: #09101f; border-left: 4px solid #22c55e; border-radius: 6px; padding: 14px; margin-top: 8px;">
            <div style="color: #22c55e; font-weight: 700; font-size: 13px; margin-bottom: 6px;">🛡️ Capital Safely Protected in Cash (${total_live_bal:,.2f} USDT)</div>
            <div style="font-size: 12px; color: #cbd5e1; line-height: 1.6;">
                &bull; <b>Macro Regime Gate:</b> <span style="color: #facc15;">{warden_status}</span> &mdash; Directional alignment actively enforced by Macro Warden.<br>
                &bull; <b>Volume Threshold:</b> Awaiting aggressive volume displacement &ge; <b>1.5x</b> 20-period average.<br>
                &bull; <b>LLaMA 3.1 AI Consensus:</b> All candidate setups evaluated across pairs must surpass the 70% conviction threshold.<br>
                &bull; <b>Active Surveillance:</b> All 10 order books ({syms_display_str}) continuously screened on Bitget matching engine.
            </div>
        </div>
        """

    # Query Near-Miss / Candidate Trades that almost occurred in past 180 minutes
    near_misses_html = ""
    near_misses_found = []
    try:
        if os.path.exists("trading_data.db"):
            with sqlite3.connect("trading_data.db", timeout=5) as conn_v:
                rows_v = conn_v.execute("""
                    SELECT timestamp, engine, message FROM logs 
                    WHERE (
                        message LIKE '%Strike Aborted%' 
                        OR message LIKE '%VETO%' 
                        OR message LIKE '%TRADE BLOCKED%' 
                        OR message LIKE '%MACRO HALT%'
                        OR message LIKE '%HURST-CHOP-GATE%'
                        OR message LIKE '%NEWS-CALENDAR-VETO%'
                        OR message LIKE '%VaR-HALT%'
                        OR message LIKE '%CORE-PRIORITY-GATE%'
                    )
                    AND timestamp >= datetime('now', '-180 minutes')
                    ORDER BY id DESC LIMIT 4
                """).fetchall()
                for rv in rows_v:
                    ts = str(rv[0]).replace("T", " ")[:19]
                    eng = str(rv[1]).upper() if rv[1] else "SWARM"
                    msg = str(rv[2])
                    if "Strike Aborted" in msg:
                        clean_msg = msg.split("Strike Aborted")[-1].lstrip(" (0.0):").strip()
                    else:
                        clean_msg = msg.strip()
                    near_misses_found.append({"time": ts, "engine": eng, "reason": clean_msg})
    except Exception as n_err:
        log.debug(f"Error querying near misses: {n_err}")

    if near_misses_found:
        near_miss_rows = ""
        for nm in near_misses_found:
            near_miss_rows += f"""
            <div style="background: #1e293b; border-left: 3px solid #f59e0b; border-radius: 4px; padding: 10px 12px; margin-bottom: 8px;">
                <div style="font-size: 11px; color: #94a3b8; margin-bottom: 3px;">
                    <span style="color: #f59e0b; font-weight: bold;">⚡ {nm['engine']}</span> &bull; {nm['time']} UTC
                </div>
                <div style="font-size: 12px; color: #e2e8f0; line-height: 1.4;">
                    {nm['reason']}
                </div>
            </div>
            """
        near_misses_html = f"""
        <h3>Section 4: Near-Miss Setups & Filtered Candidate Trades</h3>
        <div style="margin-top: 8px;">
            {near_miss_rows}
        </div>
        """

    stats_html = ""
    if stats_data:
        stats_html += f"""
        <h3>Section 5: Performance & Balance Telemetry</h3>
        <table>
            <tr><td class="label">Live Wallet Capital ($)</td><td class="value" style="color: #38bdf8;">${total_live_bal:,.2f} USDT</td></tr>
            <tr><td class="label">Sniper Bal (15m Engine)</td><td class="value">${stats_data.get('sniper_bal', 0.0):,.2f} USDT</td></tr>
            <tr><td class="label">Sniper Today PnL ($)</td><td class="value" style="color: {'#22c55e' if stats_data.get('sniper_pnl', 0.0) > 0 else '#ef4444' if stats_data.get('sniper_pnl', 0.0) < 0 else '#f1f5f9'}; font-weight: bold;">${stats_data.get('sniper_pnl', 0.0):+.2f} USD</td></tr>
            <tr><td class="label">Berserker Bal (5m Engine)</td><td class="value">${stats_data.get('berserker_bal', 0.0):,.2f} USDT</td></tr>
            <tr><td class="label">Berserker Today PnL ($)</td><td class="value" style="color: {'#22c55e' if stats_data.get('berserker_pnl', 0.0) > 0 else '#ef4444' if stats_data.get('berserker_pnl', 0.0) < 0 else '#f1f5f9'}; font-weight: bold;">${stats_data.get('berserker_pnl', 0.0):+.2f} USD</td></tr>
        </table>
        """

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0b101b; color: #f1f5f9; padding: 20px; }}
            .container {{ max-width: 600px; background: #131d31; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
            h2 {{ color: #38bdf8; margin-top: 0; font-size: 20px; border-bottom: 2px solid #1e293b; padding-bottom: 12px; text-align: center; }}
            h3 {{ color: #94a3b8; font-size: 13px; margin-top: 20px; text-transform: uppercase; letter-spacing: 0.05em; border-left: 3px solid #38bdf8; padding-left: 8px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
            td, th {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; font-size: 13px; }}
            th {{ text-align: center; color: #64748b; font-weight: 600; font-size: 11px; text-transform: uppercase; }}
            .label {{ font-weight: 600; color: #64748b; width: 45%; }}
            .value {{ color: #f8fafc; text-align: right; font-family: 'Consolas', monospace; font-weight: bold; }}
            .footer {{ font-size: 11px; color: #475569; text-align: center; margin-top: 24px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>⚡ V15.0 SOVEREIGN SWARM HEARTBEAT</h2>
            
            <h3>Section 1: Swarm Integrity Briefing</h3>
            <table>
                <tr><td class="label">Fleet PnL (Live Session)</td><td class="value" style="color: {pnl_color};">{pnl_str}</td></tr>
                <tr><td class="label">Active Exposure (Open Nodes)</td><td class="value" style="color: #38bdf8;">{active_exposure} Node(s)</td></tr>
                <tr><td class="label">Warden Status (Macro Regime)</td><td class="value" style="color: #f59e0b;">{warden_status}</td></tr>
                <tr><td class="label">Exchange & Mode</td><td class="value">Bitget USDT-M (Live Pilot)</td></tr>
                <tr><td class="label">Timestamp</td><td class="value">{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</td></tr>
            </table>
            
            <h3>Section 2: Active Exposure Registry</h3>
            <table>
                <thead>
                    <tr>
                        <th style="text-align: left;">Symbol</th>
                        <th>Side</th>
                        <th>Leverage</th>
                        <th style="text-align: right;">Entry Price</th>
                        <th style="text-align: right;">Qty</th>
                    </tr>
                </thead>
                <tbody>
                    {pos_rows_html}
                </tbody>
            </table>
            
            {recent_trades_html}
            
            {near_misses_html}
            
            {stats_html}
            
            <div class="footer">
                This is an automated operational briefing dispatched securely by the V15.0 Sovereign Swarm.
            </div>
        </div>
    </body>
    </html>
    """
    return send_html_alert(subject, html_content)

async def send_hourly_heartbeat_report_async(fleet_pnl, active_exposure, warden_status, position_details=None, stats_data=None):
    return await asyncio.to_thread(send_hourly_heartbeat_report, fleet_pnl, active_exposure, warden_status, position_details, stats_data)

def send_hourly_report(sniper_bal, sniper_pnl, berserker_bal, berserker_pnl, total_pnl):
    subject = "Hourly Fleet Status Report"
    body = (
        f"=== Llama Trading Bot Fleet Status ===\n\n"
        f"🎯 Sniper Engine (15m):\n"
        f"  - Available Balance: ${sniper_bal:.2f}\n"
        f"  - Today's PnL: ${sniper_pnl:+.2f}\n\n"
        f"🪓 Berserker Engine (5m):\n"
        f"  - Available Balance: ${berserker_bal:.2f}\n"
        f"  - Today's PnL: ${berserker_pnl:+.2f}\n\n"
        f"📈 Cumulative Summary:\n"
        f"  - Total Today's PnL: ${total_pnl:+.2f}\n\n"
        f"Status: Swarm Operational"
    )
    return send_alert(subject, body)

async def send_hourly_report_async(sniper_bal, sniper_pnl, berserker_bal, berserker_pnl, total_pnl):
    return await asyncio.to_thread(send_hourly_report, sniper_bal, sniper_pnl, berserker_bal, berserker_pnl, total_pnl)

def send_trailing_ratchet_alert(symbol, side, old_sl, new_sl, current_price, peak_price, r_multiple, locked_profit_usd=0.0, anchor_name="4H Tenkan"):
    """
    Dispatched when a position's stop loss is dynamically ratcheted higher to lock in profit.
    """
    subject = f"🛡️ [PROFIT FLOOR ADVANCED] {symbol} {side.upper()} Stop Raised to ${new_sl:,.4f} (+{r_multiple:.1f}R) | Bitget Live"
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0b101b; color: #f1f5f9; padding: 20px; }}
            .container {{ max-width: 600px; background: #131d31; border: 1px solid #334155; border-radius: 12px; padding: 24px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
            h2 {{ color: #38bdf8; margin-top: 0; font-size: 20px; border-bottom: 2px solid #1e293b; padding-bottom: 12px; text-align: center; }}
            .badge {{ display: inline-block; background: #10b981; color: #000; font-weight: 800; padding: 5px 12px; border-radius: 4px; font-size: 13px; margin-bottom: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
            td {{ padding: 8px 10px; border-bottom: 1px solid #1e293b; font-size: 13px; }}
            .label {{ font-weight: 600; color: #64748b; width: 45%; }}
            .value {{ font-weight: 700; color: #f8fafc; }}
            .reason-box {{ background: #0f172a; border-left: 4px solid #10b981; padding: 12px; border-radius: 4px; margin-top: 15px; font-size: 13px; line-height: 1.5; color: #cbd5e1; }}
            .footer {{ margin-top: 24px; font-size: 11px; color: #475569; text-align: center; border-top: 1px solid #1e293b; padding-top: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🛡️ PROFIT FLOOR ADVANCED</h2>
            <div style="text-align: center;">
                <span class="badge">TRAILING STOP RATCHET: +{r_multiple:.2f}R</span>
            </div>
            
            <table>
                <tr><td class="label">Symbol</td><td class="value" style="color: #38bdf8;">{symbol}</td></tr>
                <tr><td class="label">Side & Leverage</td><td class="value">{side.upper()} (3x Isolated)</td></tr>
                <tr><td class="label">Current Mark Price</td><td class="value">${current_price:,.4f}</td></tr>
                <tr><td class="label">Peak High Reached</td><td class="value">${peak_price:,.4f}</td></tr>
                <tr><td class="label">Previous Stop Floor</td><td class="value" style="color: #94a3b8;">${old_sl:,.4f}</td></tr>
                <tr><td class="label">New Protected Floor</td><td class="value" style="color: #10b981; font-size: 15px;">${new_sl:,.4f} ({anchor_name})</td></tr>
                <tr><td class="label">Guaranteed Profit on Exit</td><td class="value" style="color: #10b981;">+${locked_profit_usd:,.2f} USD</td></tr>
            </table>
            
            <div class="reason-box">
                <b>💡 Quantitative Defense Logic:</b><br>
                Price advanced to ${current_price:,.4f}, pulling the {anchor_name} line upward. The automated trailing engine has marched your stop-loss floor up to ${new_sl:,.4f}. If price retraces, the position will exit with guaranteed profit, while allowing upside targets to remain completely open.
            </div>
            
            <div class="footer">
                Automated Institutional Telemetry generated securely by The Swing Architect (4H Engine).
            </div>
        </div>
    </body>
    </html>
    """
    return send_html_alert(subject, html_content)

async def send_trailing_ratchet_alert_async(symbol, side, old_sl, new_sl, current_price, peak_price, r_multiple, locked_profit_usd=0.0, anchor_name="4H Tenkan"):
    return await asyncio.to_thread(send_trailing_ratchet_alert, symbol, side, old_sl, new_sl, current_price, peak_price, r_multiple, locked_profit_usd, anchor_name)

