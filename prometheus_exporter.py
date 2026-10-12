import sqlite3
import time
import datetime
import os
import logging
import psutil
from prometheus_client import start_http_server, Gauge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger('prometheus_exporter')

# Define Prometheus Metrics
# Capital and Risk Exposure
metric_swarm_capital = Gauge('swarm_capital_usdt', 'Current capital balance of the swarm agent', ['agent'])
metric_swarm_positions = Gauge('swarm_open_positions_count', 'Total number of active open positions', ['agent'])
metric_swarm_exposure = Gauge('swarm_latent_exposure_usdt', 'Total nominal size of active open positions', ['agent'])
metric_swarm_win_rate = Gauge('swarm_win_rate_percent', 'Historical win rate percentage', ['agent'])
metric_swarm_trades = Gauge('swarm_trades_total', 'Total historical trades executed', ['agent'])
metric_swarm_volume = Gauge('swarm_trade_volume_usdt_total', 'Total historical trading volume', ['agent'])
metric_swarm_confidence = Gauge('swarm_action_confidence', 'Average AI confidence of open positions', ['agent'])

# System Health
metric_sys_cpu = Gauge('system_cpu_usage_percent', 'System CPU usage percentage')
metric_sys_mem = Gauge('system_memory_usage_percent', 'System memory usage percentage')
metric_db_size = Gauge('system_db_size_bytes', 'Size of SQLite database files in bytes', ['db_name'])
metric_swarm_daily_pnl = Gauge('swarm_daily_realized_pnl_usdt', 'Rolling 24h realized PnL', ['agent'])

# Advanced Risk
metric_swarm_drawdown_percent = Gauge('swarm_drawdown_percent', 'Current drawdown from peak capital', ['agent'])
metric_emergency_brake = Gauge('system_emergency_brake_active', '1 if Flash-Crash veto is triggered, else 0')
metric_swarm_vetoes = Gauge('swarm_agent_veto_active', '1 if the agent has an active trade veto, else 0', ['agent'])
metric_swarm_veto_count = Gauge('swarm_active_vetoes_total', 'Total count of currently active agent vetoes')

# State variables for Drawdown calculation
max_balances = {
    'sniper': 0.0,
    'berserker': 0.0,
    'marl': 0.0,
    'mean_reversion': 0.0
}

# Market Regime
metric_hurst = Gauge('market_regime_hurst', 'Current Hurst Exponent (Trend vs Mean-Reverting)')
metric_adx = Gauge('market_regime_adx', 'Current Average Directional Index (Trend Strength)')

# Sentiment
metric_fear_score = Gauge('sentiment_fear_score', 'Current market fear index score')

# Oracle
metric_vwap = Gauge('oracle_vwap', 'Volume Weighted Average Price from Oracle')

def table_exists(cursor, table_name):
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cursor.fetchone() is not None
    except Exception:
        return False

def read_system_state(cursor, key, default="0.0"):
    if not table_exists(cursor, "system_state"):
        return default
    try:
        cursor.execute("SELECT value FROM system_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            # support string formats if key contains strings like last_veto
            val = row[0]
            try:
                return float(val)
            except ValueError:
                return val
    except Exception as e:
        log.warning(f"Failed to read system_state key {key}: {e}")
    return default

def read_latest_regime(cursor):
    if not table_exists(cursor, "market_regime"):
        return 0.0, 0.0
    try:
        cursor.execute("SELECT hurst, adx FROM market_regime ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return float(row[0] or 0), float(row[1] or 0)
    except Exception as e:
        log.warning(f"Failed to read market_regime: {e}")
    return 0.0, 0.0

def read_latest_sentiment(cursor):
    if not table_exists(cursor, "sentiment_logs"):
        return 50.0  # Neutral fallback when no sentiment logged
    try:
        cursor.execute("SELECT score FROM sentiment_logs ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as e:
        log.warning(f"Failed to read sentiment_logs: {e}")
    return 50.0

def read_latest_oracle(cursor):
    if not table_exists(cursor, "oracle_state"):
        return 0.0
    try:
        cursor.execute("SELECT vwap FROM oracle_state ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            return float(row[0] or 0)
    except Exception as e:
        log.warning(f"Failed to read oracle_state: {e}")
    return 0.0

def read_positions_stats(cursor, table_name):
    if not table_exists(cursor, table_name):
        return 0, 0.0
    try:
        cursor.execute(f"SELECT COUNT(*), SUM(entry_price * qty) FROM {table_name}")
        row = cursor.fetchone()
        if row:
            count = int(row[0] or 0)
            exposure = float(row[1] or 0.0)
            return count, exposure
    except Exception as e:
        log.warning(f"Failed to read {table_name} stats: {e}")
    return 0, 0.0

def read_trade_stats(cursor, table_name):
    if not table_exists(cursor, table_name):
        return 0, 0.0, 0.0
    try:
        cursor.execute(f"SELECT COUNT(*), SUM(entry_price * qty) FROM {table_name}")
        row = cursor.fetchone()
        total_trades = int(row[0] or 0)
        total_volume = float(row[1] or 0.0)
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE pnl > 0")
        wins = int(cursor.fetchone()[0] or 0)
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE pnl < 0")
        losses = int(cursor.fetchone()[0] or 0)
        
        win_rate = 0.0
        if (wins + losses) > 0:
            win_rate = (wins / (wins + losses)) * 100.0
            
        return total_trades, total_volume, win_rate
    except Exception as e:
        log.warning(f"Failed to read {table_name} trade stats: {e}")
    return 0, 0.0, 0.0

def read_confidence_stats(cursor, table_name):
    if not table_exists(cursor, table_name):
        return 0.0
    try:
        cursor.execute(f"SELECT AVG(action_confidence) FROM {table_name}")
        row = cursor.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as e:
        log.warning(f"Failed to read {table_name} confidence stats: {e}")
    return 0.0

def read_daily_pnl(cursor, table_name):
    if not table_exists(cursor, table_name):
        return 0.0
    try:
        past_24h = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).isoformat()
        cursor.execute(f"SELECT SUM(pnl) FROM {table_name} WHERE timestamp > ?", (past_24h,))
        row = cursor.fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as e:
        log.warning(f"Failed to read {table_name} daily pnl: {e}")
    return 0.0

def get_db_size(db_path):
    size = 0
    if os.path.exists(db_path):
        size += os.path.getsize(db_path)
    wal_path = db_path + "-wal"
    if os.path.exists(wal_path):
        size += os.path.getsize(wal_path)
    return size

def collect_metrics(data_db_path, ledger_db_path):
    try:
        # Trading Data DB (System State, Regime)
        with sqlite3.connect(data_db_path, timeout=15.0) as conn_data:
            conn_data.execute("PRAGMA journal_mode=WAL;")
            conn_data.execute("PRAGMA busy_timeout=15000;")
            c_data = conn_data.cursor()
            
            # Capital Balances
            balance_sniper = read_system_state(c_data, 'balance_sniper', 0.0)
            balance_berserker = read_system_state(c_data, 'balance_berserker', 0.0)
            balance_marl = read_system_state(c_data, 'balance_marl', 0.0)
            balance_mean_reversion = read_system_state(c_data, 'balance_mean_reversion', 0.0)
            
            metric_swarm_capital.labels(agent='sniper').set(balance_sniper)
            metric_swarm_capital.labels(agent='berserker').set(balance_berserker)
            metric_swarm_capital.labels(agent='marl').set(balance_marl)
            metric_swarm_capital.labels(agent='mean_reversion').set(balance_mean_reversion)
            
            # Drawdown Calculation (Read-Only Peak Balance Tracking)
            global max_balances
            for agent in ['sniper', 'berserker', 'marl', 'mean_reversion']:
                peak_key = f"peak_balance_{agent}"
                current_bal = locals()[f"balance_{agent}"]
                stored_peak = read_system_state(c_data, peak_key, 0.0)
                # Keep in-memory peak without performing write operations in the exporter
                peak = max(stored_peak, max_balances.get(agent, 0.0), current_bal)
                max_balances[agent] = peak
            
            dd_sniper = ((max_balances['sniper'] - balance_sniper) / max_balances['sniper'] * 100) if max_balances['sniper'] > 0 else 0
            dd_berserker = ((max_balances['berserker'] - balance_berserker) / max_balances['berserker'] * 100) if max_balances['berserker'] > 0 else 0
            dd_marl = ((max_balances['marl'] - balance_marl) / max_balances['marl'] * 100) if max_balances['marl'] > 0 else 0
            dd_mean_reversion = ((max_balances['mean_reversion'] - balance_mean_reversion) / max_balances['mean_reversion'] * 100) if max_balances['mean_reversion'] > 0 else 0
            
            metric_swarm_drawdown_percent.labels(agent='sniper').set(dd_sniper)
            metric_swarm_drawdown_percent.labels(agent='berserker').set(dd_berserker)
            metric_swarm_drawdown_percent.labels(agent='marl').set(dd_marl)
            metric_swarm_drawdown_percent.labels(agent='mean_reversion').set(dd_mean_reversion)
            
            # Position & Risk Exposure Metrics
            sniper_count, sniper_exp = read_positions_stats(c_data, 'open_positions')
            berserker_count, berserker_exp = read_positions_stats(c_data, 'berserker_open_positions')
            marl_count, marl_exp = read_positions_stats(c_data, 'marl_live_open_positions')
            mr_count, mr_exp = read_positions_stats(c_data, 'mean_reversion_open_positions')
            
            metric_swarm_positions.labels(agent='sniper').set(sniper_count)
            metric_swarm_positions.labels(agent='berserker').set(berserker_count)
            metric_swarm_positions.labels(agent='marl').set(marl_count)
            metric_swarm_positions.labels(agent='mean_reversion').set(mr_count)
            
            metric_swarm_exposure.labels(agent='sniper').set(sniper_exp)
            metric_swarm_exposure.labels(agent='berserker').set(berserker_exp)
            metric_swarm_exposure.labels(agent='marl').set(marl_exp)
            metric_swarm_exposure.labels(agent='mean_reversion').set(mr_exp)
            
            # Confidence Metrics
            metric_swarm_confidence.labels(agent='sniper').set(read_confidence_stats(c_data, 'open_positions'))
            metric_swarm_confidence.labels(agent='berserker').set(read_confidence_stats(c_data, 'berserker_open_positions'))
            metric_swarm_confidence.labels(agent='marl').set(read_confidence_stats(c_data, 'marl_live_open_positions'))
            metric_swarm_confidence.labels(agent='mean_reversion').set(read_confidence_stats(c_data, 'mean_reversion_open_positions'))
            
            # Trade Stats
            s_trades, s_vol, s_win = read_trade_stats(c_data, 'trades')
            b_trades, b_vol, b_win = read_trade_stats(c_data, 'berserker_trades')
            m_trades, m_vol, m_win = read_trade_stats(c_data, 'marl_live_trades')
            mr_trades, mr_vol, mr_win = read_trade_stats(c_data, 'mean_reversion_trades')
            
            metric_swarm_trades.labels(agent='sniper').set(s_trades)
            metric_swarm_trades.labels(agent='berserker').set(b_trades)
            metric_swarm_trades.labels(agent='marl').set(m_trades)
            metric_swarm_trades.labels(agent='mean_reversion').set(mr_trades)
            
            metric_swarm_volume.labels(agent='sniper').set(s_vol)
            metric_swarm_volume.labels(agent='berserker').set(b_vol)
            metric_swarm_volume.labels(agent='marl').set(m_vol)
            metric_swarm_volume.labels(agent='mean_reversion').set(mr_vol)
            
            metric_swarm_win_rate.labels(agent='sniper').set(s_win)
            metric_swarm_win_rate.labels(agent='berserker').set(b_win)
            metric_swarm_win_rate.labels(agent='marl').set(m_win)
            metric_swarm_win_rate.labels(agent='mean_reversion').set(mr_win)
            
            # Daily PnL
            metric_swarm_daily_pnl.labels(agent='sniper').set(read_daily_pnl(c_data, 'trades'))
            metric_swarm_daily_pnl.labels(agent='berserker').set(read_daily_pnl(c_data, 'berserker_trades'))
            metric_swarm_daily_pnl.labels(agent='marl').set(read_daily_pnl(c_data, 'marl_live_trades'))
            metric_swarm_daily_pnl.labels(agent='mean_reversion').set(read_daily_pnl(c_data, 'mean_reversion_trades'))
            
            # Regime Metrics
            hurst, adx = read_latest_regime(c_data)
            metric_hurst.set(hurst)
            metric_adx.set(adx)
            
            # Veto Metrics
            active_vetoes = 0
            for agent in ['sniper', 'berserker', 'marl', 'mean_reversion']:
                veto_val = read_system_state(c_data, f"last_veto_{agent}", "ACTIVE")
                is_vetoed = 1 if (veto_val != "ACTIVE" and veto_val != "0" and veto_val != "") else 0
                metric_swarm_vetoes.labels(agent=agent).set(is_vetoed)
                if is_vetoed:
                    active_vetoes += 1
            metric_swarm_veto_count.set(active_vetoes)
            
        # Trading Ledger DB (Sentiment, Oracle)
        with sqlite3.connect(ledger_db_path, timeout=15.0) as conn_ledger:
            conn_ledger.execute("PRAGMA journal_mode=WAL;")
            conn_ledger.execute("PRAGMA busy_timeout=15000;")
            c_ledger = conn_ledger.cursor()
            
            # Sentiment
            fear_score = read_latest_sentiment(c_ledger)
            metric_fear_score.set(fear_score)
            
            # Oracle VWAP
            vwap = read_latest_oracle(c_ledger)
            metric_vwap.set(vwap)
            
            # Emergency Brake State
            spy_health = 0.0
            try:
                c_ledger.execute("SELECT spy_health FROM macro_logs ORDER BY id DESC LIMIT 1")
                row = c_ledger.fetchone()
                if row:
                    spy_health = float(row[0] or 0.0)
            except Exception:
                pass
            
            brake_active = 1 if (fear_score < 25.0 or spy_health < -0.6) else 0
            metric_emergency_brake.set(brake_active)
            
        # System Health
        metric_sys_cpu.set(psutil.cpu_percent())
        metric_sys_mem.set(psutil.virtual_memory().percent)
        metric_db_size.labels(db_name='trading_data').set(get_db_size(data_db_path))
        metric_db_size.labels(db_name='trading_ledger').set(get_db_size(ledger_db_path))

        log.info("Successfully polled metrics from SQLite ledgers.")
    except Exception as e:
        log.error(f"Error during metric collection: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prometheus Exporter for Llama Trading Bot")
    parser.add_argument('--port', type=int, default=8000, help='Port to expose metrics')
    parser.add_argument('--data-db', type=str, default='trading_data.db', help='Path to trading_data.db')
    parser.add_argument('--ledger-db', type=str, default='trading_ledger.db', help='Path to trading_ledger.db')
    parser.add_argument('--interval', type=int, default=15, help='Polling interval in seconds')
    args = parser.parse_args()
    
    # Resolve absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, args.data_db)
    ledger_path = os.path.join(base_dir, args.ledger_db)
    
    start_http_server(args.port)
    log.info(f"Prometheus Exporter started on port {args.port}")
    log.info(f"Monitoring: {data_path} & {ledger_path}")
    
    try:
        while True:
            collect_metrics(data_path, ledger_path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Exporter shutting down.")
