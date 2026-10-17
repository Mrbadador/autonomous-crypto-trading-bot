import os
import logging
from dotenv import load_dotenv

# [P0-FIX] Anchor .env load to this file's directory — CWD-independent.
# A bare load_dotenv() resolves relative to the process CWD, which can drift
# in subprocesses (e.g. Streamlit changes CWD), causing BINANCE_API_KEY to
# load as an empty string and every signed request to return -1021 / -2014.
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_HERE, ".env"), override=True)

_log = logging.getLogger("config")

# ---------------------------------------------------------------------------
# Engine Mode
# ---------------------------------------------------------------------------
MODE             = os.getenv("MODE", "live_pilot").lower()
PAPER_TRADE      = os.getenv("PAPER_TRADE", "false").lower() in ("true", "1")
IS_REPLAY        = (MODE == "replay")
DATA_DIR         = os.getenv("DATA_DIR", "data/historical")
REPLAY_DELAY     = float(os.getenv("REPLAY_DELAY", "0.0"))
BYPASS_AI_IN_REPLAY = os.getenv("BYPASS_AI_IN_REPLAY", "True").lower() in ("true", "1")
REPLAY_START_DATE = os.getenv("REPLAY_START_DATE", "2023-04-01")
REPLAY_END_DATE   = os.getenv("REPLAY_END_DATE", "2023-07-31")

# ---------------------------------------------------------------------------
# Exchange / Symbol
# ---------------------------------------------------------------------------
SANDBOX          = os.getenv("SANDBOX", "true").lower() in ("true", "1")
SYMBOLS_STR      = os.getenv("SYMBOLS") or "BTC/USDT"
def normalize_symbol(s):
    if not s: return ""
    s = s.strip().upper()
    if "/" not in s and s.endswith("USDT"):
        return s.replace("USDT", "/USDT")
    return s

SYMBOLS = [normalize_symbol(s) for s in SYMBOLS_STR.split(",") if s and s.strip()]

# ---------------------------------------------------------------------------
# API Keys (Loaded securely from .env)
# ---------------------------------------------------------------------------
EXCHANGE_ID        = os.getenv("EXCHANGE_ID", "bitget").lower()
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "") or os.getenv("BINANCE_API_SECRET", "")
BITGET_API_KEY     = os.getenv("BITGET_API_KEY", "")
BITGET_API_SECRET  = os.getenv("BITGET_API_SECRET", "")
BITGET_PASSPHRASE  = os.getenv("BITGET_PASSPHRASE", "") or os.getenv("API_PASSWORD", "")

ACTIVE_API_KEY     = BITGET_API_KEY if EXCHANGE_ID == "bitget" else BINANCE_API_KEY
ACTIVE_SECRET_KEY  = BITGET_API_SECRET if EXCHANGE_ID == "bitget" else BINANCE_SECRET_KEY
ACTIVE_PASSPHRASE  = BITGET_PASSPHRASE if EXCHANGE_ID == "bitget" else ""

# ---------------------------------------------------------------------------
# Indicator Periods
# ---------------------------------------------------------------------------
EMA_PERIOD                    = 20
RSI_PERIOD                   = 14
ATR_PERIOD                   = 14
ADX_WINDOW                   = 100   # Rolling window for dynamic ADX thresholds
VOLATILITY_BASELINE          = 0.005 # Baseline for normalizing volatility state

# ---------------------------------------------------------------------------
# Capital & Leverage
# ---------------------------------------------------------------------------
INITIAL_BALANCE              = float(os.getenv("INITIAL_BALANCE", "1000.0"))
RISK_PER_TRADE_PCT           = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))  # Default 0.5% risk per trade (institutional capital preservation)
LEVERAGE                     = int(os.getenv("LEVERAGE", "5"))
MARGIN_MODE                  = os.getenv("MARGIN_MODE", "isolated").lower()
TRADE_FEE_PERCENT            = float(os.getenv("TRADE_FEE_PERCENT", "0.0004"))
FEE_RATE                     = TRADE_FEE_PERCENT
FUNDING_RATE_PERCENT         = 0.0001   # 0.01% per 8h

# ---------------------------------------------------------------------------
# BTC Cash-and-Carry Basis Trader (Delta-Neutral Funding Capture)
# ---------------------------------------------------------------------------
BASIS_TRADER_ENABLED          = os.getenv("BASIS_TRADER_ENABLED", "false").lower() in ("true", "1")
BASIS_TRADE_ALLOCATION_USDT   = float(os.getenv("BASIS_TRADE_ALLOCATION_USDT", "1000.0"))
BASIS_MIN_ANNUAL_YIELD        = float(os.getenv("BASIS_MIN_ANNUAL_YIELD", "0.08"))   # 8.0% min annualized hurdle
BASIS_UNWIND_ANNUAL_YIELD     = float(os.getenv("BASIS_UNWIND_ANNUAL_YIELD", "0.015")) # 1.5% unwind threshold
BASIS_MAX_DISLOCATION_PCT     = float(os.getenv("BASIS_MAX_DISLOCATION_PCT", "0.015")) # 1.5% spot-perp divergence guard
BASIS_CHECK_INTERVAL_SECS     = int(os.getenv("BASIS_CHECK_INTERVAL_SECS", "30"))
BASIS_MIN_HOLDING_DAYS        = int(os.getenv("BASIS_MIN_HOLDING_DAYS", "7"))        # 7 days min holding anti-churn
BASIS_ESTIMATED_ROUNDTRIP_FEE_BPS = float(os.getenv("BASIS_ESTIMATED_ROUNDTRIP_FEE_BPS", "0.0024")) # 24 bps fee drag
BASIS_MARGIN_MODE             = os.getenv("BASIS_MARGIN_MODE", "crossed").lower()    # UTA cross-margin buffer
BASIS_MIN_LIQ_DISTANCE_PCT    = float(os.getenv("BASIS_MIN_LIQ_DISTANCE_PCT", "0.20")) # 20% liquidation threshold
BASIS_PERP_FILL_TIMEOUT_SECS  = int(os.getenv("BASIS_PERP_FILL_TIMEOUT_SECS", "15"))  # 15s sequential fill timeout

# ---------------------------------------------------------------------------
# Execution Fleet Architecture Switch
# ---------------------------------------------------------------------------
# Swing-Only Consolidation: Deactivates high-frequency 5m/15m scalpers to eliminate taker fee churn.
SCALPERS_ENABLED              = os.getenv("SCALPERS_ENABLED", "false").lower() in ("true", "1")

# ---------------------------------------------------------------------------
# The Super Bot (Unified Dual-Regime Engine) & Swing Architect
# ---------------------------------------------------------------------------
SUPER_BOT_ENABLED             = os.getenv("SUPER_BOT_ENABLED", "true").lower() in ("true", "1")
# Maximum age of the market_regime row for opening NEW positions. The classifier republishes
# Database Integrity & WAL Maintenance Intervals
DB_INTEGRITY_CHECKPOINT_INTERVAL_SECS = int(os.getenv("DB_INTEGRITY_CHECKPOINT_INTERVAL_SECS", "300"))
DB_INTEGRITY_MAX_AGE_SEC              = float(os.getenv("DB_INTEGRITY_MAX_AGE_SEC", str(DB_INTEGRITY_CHECKPOINT_INTERVAL_SECS * 3)))

REGIME_MAX_AGE_SEC            = float(os.getenv("REGIME_MAX_AGE_SEC", "900"))
GRID_LEVERAGE                 = int(os.getenv("GRID_LEVERAGE", "3"))
GRID_ALLOCATION_USDT          = float(os.getenv("GRID_ALLOCATION_USDT", "85.0"))  # Proportional Sizing: $85 margin per trade
GRID_TP_PCT                   = float(os.getenv("GRID_TP_PCT", "0.0135"))         # +1.35% limit target (1.17 R:R vs 1.15% SL)
GRID_SL_PCT                   = float(os.getenv("GRID_SL_PCT", "0.0115"))         # -1.15% tight stop
GRID_RSI_THRESHOLD            = float(os.getenv("GRID_RSI_THRESHOLD", "34.0"))  # Relaxed to 34.0 to catch standard 5m pullbacks

# ---------------------------------------------------------------------------
# Quick Scalp Short Engine (Fast Turnover Protocol - Disabled: Restored Long Swing + Maker Grid Core)
# ---------------------------------------------------------------------------
SCALP_SHORT_ENABLED            = os.getenv("SCALP_SHORT_ENABLED", "false").lower() in ("true", "1")
SCALP_SHORT_TP_PCT             = float(os.getenv("SCALP_SHORT_TP_PCT", "0.0120"))     # +1.20% limit target (1.20 R:R vs 1.00% SL)
SCALP_SHORT_SL_PCT             = float(os.getenv("SCALP_SHORT_SL_PCT", "0.0100"))     # -1.00% tight stop
SCALP_SHORT_BE_TRIGGER_PCT     = float(os.getenv("SCALP_SHORT_BE_TRIGGER_PCT", "0.0050")) # +0.50% gain triggers breakeven ratchet
SCALP_SHORT_BE_FEE_BUFFER_PCT  = float(os.getenv("SCALP_SHORT_BE_FEE_BUFFER_PCT", "0.0018")) # 0.18% fee cushion (covers 2x taker fees + slippage)
SCALP_SHORT_MAX_HOLD_MINS      = float(os.getenv("SCALP_SHORT_MAX_HOLD_MINS", "45.0")) # 45 minutes soft profit harvest / stop-tighten checkpoint
SCALP_SHORT_HARD_EXPIRY_MINS   = float(os.getenv("SCALP_SHORT_HARD_EXPIRY_MINS", "90.0")) # 90 minutes hard time decay limit
SCALP_SHORT_RSI_OVERBOUGHT     = float(os.getenv("SCALP_SHORT_RSI_OVERBOUGHT", "68.0")) # Chop range top RSI floor
SCALP_SHORT_MIN_RSI            = float(os.getenv("SCALP_SHORT_MIN_RSI", "38.0"))      # Strict anti-bottom filter (never short < 38)
MAX_CONCURRENT_SHORTS          = int(os.getenv("MAX_CONCURRENT_SHORTS", "1"))         # Strictly max 1 concurrent short to prevent correlated herd wipeouts
SCALP_SHORT_BTC_PUMP_GUARD     = os.getenv("SCALP_SHORT_BTC_PUMP_GUARD", "true").lower() in ("true", "1") # Veto altcoin shorts if BTC is in 5m/15m bullish push
SCALP_SHORT_REQUIRE_15M_EXHAUST = os.getenv("SCALP_SHORT_REQUIRE_15M_EXHAUST", "true").lower() in ("true", "1") # Multi-timeframe 15m exhaustion requirement
SCALP_SHORT_MIN_HARVEST_PCT    = float(os.getenv("SCALP_SHORT_MIN_HARVEST_PCT", "0.0045")) # +0.45% min gain to harvest profit at 45m (guarantees net profit after fees)
SCALP_SHORT_MAX_5M_ADX         = float(os.getenv("SCALP_SHORT_MAX_5M_ADX", "22.0"))        # 5m ADX exhaustion ceiling (blocks shorting active breakouts)
SCALP_SHORT_MAX_24H_GAIN_PCT   = float(os.getenv("SCALP_SHORT_MAX_24H_GAIN_PCT", "3.0"))   # Never short an asset up > +3.0% on 24h in CHOP mode
SCALP_SHORT_MAX_DI_SPREAD      = float(os.getenv("SCALP_SHORT_MAX_DI_SPREAD", "4.0"))      # Mute CHOP shorts when market +DI exceeds -DI by > 4.0

SWING_TRADER_ENABLED          = os.getenv("SWING_TRADER_ENABLED", "true").lower() in ("true", "1")
SWING_MAX_CONCURRENT_LOCKS    = int(os.getenv("SWING_MAX_CONCURRENT_LOCKS", "4"))
SWING_TRADE_ALLOCATION_USDT   = float(os.getenv("SWING_TRADE_ALLOCATION_USDT", "85.0"))  # Proportional Sizing: $85 margin per trade (~17.5% of wallet)
SWING_BALANCE_PCT             = float(os.getenv("SWING_BALANCE_PCT", "0.175"))  # 17.5% per trade (4 concurrent locks = 70.0% total wallet margin)
SWING_LEVERAGE                = int(os.getenv("SWING_LEVERAGE", "3"))
SWING_STANDARD_LEVERAGE       = int(os.getenv("SWING_STANDARD_LEVERAGE", "3"))
SWING_HIGH_QUALITY_LEVERAGE   = int(os.getenv("SWING_HIGH_QUALITY_LEVERAGE", "5"))
SWING_HIGH_QUALITY_CONVICTION_MIN = float(os.getenv("SWING_HIGH_QUALITY_CONVICTION_MIN", "0.85"))
SWING_HIGH_QUALITY_ADX_MIN    = float(os.getenv("SWING_HIGH_QUALITY_ADX_MIN", "32.0"))
SWING_ATR_SL_MULT             = float(os.getenv("SWING_ATR_SL_MULT", "3.0"))
SWING_RUNNER_TP_MULT          = float(os.getenv("SWING_RUNNER_TP_MULT", "4.0"))
SWING_BREAKEVEN_R_THRESHOLD   = float(os.getenv("SWING_BREAKEVEN_R_THRESHOLD", "2.0"))  # Relaxed to 2.0R to let 4H swings breathe
SWING_MIN_ADX                 = float(os.getenv("SWING_MIN_ADX", "22.0"))
SWING_CHECK_INTERVAL_SECS     = int(os.getenv("SWING_CHECK_INTERVAL_SECS", "300"))
# Mega-Cap Segregation: Excludes BTC, ETH, and BNB from 4H breakout trend engine to eliminate -$13.58 drag
SWING_BLACKLIST               = [s.strip() for s in os.getenv("SWING_BLACKLIST", "DOGE/USDT,BTC/USDT,ETH/USDT,BNB/USDT").split(",") if s.strip()]
SWING_PROFIT_LOCK_T1_ROE      = float(os.getenv("SWING_PROFIT_LOCK_T1_ROE", "30.0"))  # Uncage Right Tail: Lock at +30% ROE (was 15%)
SWING_PROFIT_LOCK_T1_RATIO    = float(os.getenv("SWING_PROFIT_LOCK_T1_RATIO", "0.30"))  # Floor 30% profit, let 70% run!
SWING_PROFIT_LOCK_T2_ROE      = float(os.getenv("SWING_PROFIT_LOCK_T2_ROE", "60.0"))  # Lock at +60% ROE (was 30%)
SWING_PROFIT_LOCK_T2_RATIO    = float(os.getenv("SWING_PROFIT_LOCK_T2_RATIO", "0.60"))  # Floor 60% profit, let 40% run!
SWING_MIN_SL_PCT              = float(os.getenv("SWING_MIN_SL_PCT", "0.015"))        # Rule #166: 1.5% min stop distance floor
SWING_COOLDOWN_HOURS          = float(os.getenv("SWING_COOLDOWN_HOURS", "12.0"))     # 12h post-SL cool-off
SWING_MAKER_SNIPER_TIMEOUT_SECS = int(os.getenv("SWING_MAKER_SNIPER_TIMEOUT_SECS", "180")) # Max wait for retest maker fill (3 mins)
SWING_MAKER_FALLBACK_TO_MARKET = os.getenv("SWING_MAKER_FALLBACK_TO_MARKET", "true").lower() in ("true", "1")
SWING_MAX_MARKET_CHASE_PCT    = float(os.getenv("SWING_MAX_MARKET_CHASE_PCT", "0.003"))   # Max +0.30% price advance allowed on market fallback
SWING_ABORT_COOLDOWN_HOURS    = float(os.getenv("SWING_ABORT_COOLDOWN_HOURS", "0.35"))    # 21m cool-off after unfilled sniper abort

# ---------------------------------------------------------------------------
# Quantitative Alpha Gates (Zero AI Cost — Active Pre-Boardroom Filters)
# ---------------------------------------------------------------------------
SWING_DIVERGENCE_GATE_ENABLED     = os.getenv("SWING_DIVERGENCE_GATE_ENABLED", "true").lower() in ("true", "1")
SWING_DIVERGENCE_LOOKBACK_BARS    = int(os.getenv("SWING_DIVERGENCE_LOOKBACK_BARS", "14"))

SWING_VOLUME_PROFILE_GATE_ENABLED = os.getenv("SWING_VOLUME_PROFILE_GATE_ENABLED", "true").lower() in ("true", "1")
SWING_VOLUME_PROFILE_LOOKBACK     = int(os.getenv("SWING_VOLUME_PROFILE_LOOKBACK", "48"))
SWING_VOLUME_PROFILE_VA_PCT       = float(os.getenv("SWING_VOLUME_PROFILE_VA_PCT", "0.50"))  # Focuses on core 50% Value Area
SWING_VOLUME_PROFILE_VAH_TOLERANCE = float(os.getenv("SWING_VOLUME_PROFILE_VAH_TOLERANCE", "0.985")) # Permits breakout builds within 1.5% of VAH

SWING_TD_SEQUENTIAL_GATE_ENABLED  = os.getenv("SWING_TD_SEQUENTIAL_GATE_ENABLED", "true").lower() in ("true", "1")
SWING_TD_EXHAUSTION_COUNT         = int(os.getenv("SWING_TD_EXHAUSTION_COUNT", "9"))

# Microstructure Anti-Spoofing & Cascade Veto Thresholds
HAWKES_SPOOF_VETO_THRESHOLD       = float(os.getenv("HAWKES_SPOOF_VETO_THRESHOLD", "0.78"))  # Veto only severe flash spoofing (>0.78)
HAWKES_CASCADE_RHO_THRESHOLD      = float(os.getenv("HAWKES_CASCADE_RHO_THRESHOLD", "1.05")) # Spectral radius ceiling (stationarity cutoff)
MICROSTRUCTURE_VPIN_VETO_THRESHOLD = float(os.getenv("MICROSTRUCTURE_VPIN_VETO_THRESHOLD", "0.70")) # Hard pre-trade informed flow toxicity cutoff
TRADFI_FAIL_CLOSED                = os.getenv("TRADFI_FAIL_CLOSED", "true").lower() in ("true", "1")

# Chop Grid Hyperparameters (V36.5 Quantitative Calibration)

GRID_MAX_5M_ADX                 = float(os.getenv("GRID_MAX_5M_ADX", "36.0"))           # Raised from 32.0 to 36.0 to allow moderate volatility dips
GRID_BREAKEVEN_TRIGGER_PCT      = float(os.getenv("GRID_BREAKEVEN_TRIGGER_PCT", "0.0050")) # +0.50% profit triggers micro-breakeven
GRID_BREAKEVEN_FEE_BUFFER_PCT   = float(os.getenv("GRID_BREAKEVEN_FEE_BUFFER_PCT", "0.0005")) # +0.05% above entry price (locks in net fees)

# ---------------------------------------------------------------------------
# Strategy Timeframe & Data
# ---------------------------------------------------------------------------
TIMEFRAME                    = os.getenv("TIMEFRAME", "1h")
OHLCV_LIMIT                  = int(os.getenv("OHLCV_LIMIT", "1000"))
EMA_WARMUP_PERIOD            = 400    # Candles needed for stable EMA200
MIN_CANDLES_FOR_EMA          = 250    # Absolute minimum before trading
SLEEP_SECS                   = 60

# ---------------------------------------------------------------------------
# SL / TP / Trail ATR Multipliers (Searchable hyperparameters)
# ---------------------------------------------------------------------------
STOP_LOSS_ATR_MULTIPLE       = float(os.getenv("STOP_LOSS_ATR_MULTIPLE", "2.0"))
TAKE_PROFIT_ATR_MULTIPLE     = float(os.getenv("TAKE_PROFIT_ATR_MULTIPLE", "6.0"))
TRAIL_ACTIVATION_ATR_MULTIPLE = float(os.getenv("TRAIL_ACTIVATION_ATR_MULTIPLE", "3.5"))
PARTIAL_TP_ATR_MULTIPLE       = float(os.getenv("PARTIAL_TP_ATR_MULTIPLE", "4.0"))
PARTIAL_TP_FRACTION           = 0.5   # Close 50% of position at partial TP

# ---------------------------------------------------------------------------
# Flash Crash Filter — Dual-Layer SL Architecture
# ---------------------------------------------------------------------------
# SOFT_SL_ATR_MULTIPLIER: The software-monitored stop level (same as STOP_LOSS_ATR_MULTIPLE).
#   When price breaches this level, a countdown timer starts instead of immediately closing.
# FLASH_CRASH_CONFIRMATION_SECS: How long (seconds) price must remain below the Soft SL
#   before the bot actually closes the position. A wick that recovers within this window
#   is treated as a flash crash and the position is preserved.
# HARD_SL_ATR_MULTIPLIER: The wider Bitget-native strategy order (unconditional backstop).
#   This fires immediately on the exchange regardless of the software filter.
#   Protects against genuine capitulation, black-swans, and margin liquidation risk.
SOFT_SL_ATR_MULTIPLIER       = float(os.getenv("SOFT_SL_ATR_MULTIPLIER", "2.0"))
HARD_SL_ATR_MULTIPLIER       = float(os.getenv("HARD_SL_ATR_MULTIPLIER", "3.5"))
FLASH_CRASH_CONFIRMATION_SECS = int(os.getenv("FLASH_CRASH_CONFIRMATION_SECS", "15"))  # P0-5: calibrated to 15s to eliminate cascade delay
FLASH_CRASH_MAX_BREACH_PCT   = float(os.getenv("FLASH_CRASH_MAX_BREACH_PCT", "0.003"))  # 0.30% max breach beyond Soft SL before instant abort
MTF_MIN_RUNWAY_PCT            = float(os.getenv("MTF_MIN_RUNWAY_PCT", "0.004"))          # 0.4% min runway to HTF resistance/support barrier
ALTCOIN_MIN_ADX               = float(os.getenv("ALTCOIN_MIN_ADX", "20.0"))              # 20.0 min ADX for altcoin trend breakout

# ---------------------------------------------------------------------------
# Risk Management – Replay-aware thresholds
# ---------------------------------------------------------------------------
GLOBAL_MAX_DD_LIMIT           = float(os.getenv("GLOBAL_MAX_DD_LIMIT", "99.0" if IS_REPLAY else "2.0"))
GLOBAL_MARGIN_LIMIT           = float(os.getenv("GLOBAL_MARGIN_LIMIT", "70.0"))  # 70.0% Hard Wallet Margin Ceiling (30% Cash Buffer)
MAX_MARGIN_PER_TRADE_PCT      = float(os.getenv("MAX_MARGIN_PER_TRADE_PCT", "0.20"))  # Up to 20% max margin per position
GLOBAL_MAX_POSITIONS          = int(os.getenv("GLOBAL_MAX_POSITIONS", "8"))
CORE_SYMBOLS                  = [normalize_symbol(s) for s in (os.getenv("CORE_SYMBOLS") or "BTC/USDT,ETH/USDT").split(",") if s.strip()]
MAX_ALTCOIN_POSITIONS         = int(os.getenv("MAX_ALTCOIN_POSITIONS", "6"))

# ---------------------------------------------------------------------------
# Asset Tier Sizing Multipliers & Per-Asset Leverage Matrix
# ---------------------------------------------------------------------------
# Tier 1 (Core): BTC (2x, 0.4x alloc for risk safety on small capital), ETH (5x, 1.5x alloc)
# Tier 2 (Major Alts): SOL, BNB -> 5x, 0.8x alloc
# Tier 3 (Other Alts): All other altcoins -> 5x, 0.5x alloc
ASSET_TIER_MULTIPLIERS = {
    "BTC/USDT": float(os.getenv("TIER_BTC_MULT", "0.4")),
    "ETH/USDT": float(os.getenv("TIER_ETH_MULT", "1.5")),
    "SOL/USDT": float(os.getenv("TIER_SOL_MULT", "0.8")),
    "BNB/USDT": float(os.getenv("TIER_BNB_MULT", "0.8")),
}
DEFAULT_ALTCOIN_TIER_MULT = float(os.getenv("DEFAULT_ALTCOIN_TIER_MULT", "0.5"))

def get_asset_tier_multiplier(symbol: str) -> float:
    if not symbol:
        return 1.0
    s_norm = normalize_symbol(symbol)
    if s_norm in ASSET_TIER_MULTIPLIERS:
        return ASSET_TIER_MULTIPLIERS[s_norm]
    s_clean = s_norm.replace("/", "")
    for k, v in ASSET_TIER_MULTIPLIERS.items():
        if k.replace("/", "") == s_clean:
            return v
    return DEFAULT_ALTCOIN_TIER_MULT

ASSET_LEVERAGE_MAP = {
    "BTC/USDT": int(os.getenv("BTC_LEVERAGE", "5")),
    "ETH/USDT": int(os.getenv("ETH_LEVERAGE", "5")),
    "SOL/USDT": int(os.getenv("SOL_LEVERAGE", "5")),
    "BNB/USDT": int(os.getenv("BNB_LEVERAGE", "5")),
}
DEFAULT_ALTCOIN_LEVERAGE = int(os.getenv("DEFAULT_ALTCOIN_LEVERAGE", "5"))

def get_asset_leverage(symbol: str) -> int:
    if not symbol:
        return DEFAULT_ALTCOIN_LEVERAGE
    s_norm = normalize_symbol(symbol)
    if s_norm in ASSET_LEVERAGE_MAP:
        return ASSET_LEVERAGE_MAP[s_norm]
    s_clean = s_norm.replace("/", "")
    for k, v in ASSET_LEVERAGE_MAP.items():
        if k.replace("/", "") == s_clean:
            return v
    return DEFAULT_ALTCOIN_LEVERAGE
CHOP_LEVERAGE                 = int(os.getenv("CHOP_LEVERAGE", "3"))
TREND_LEVERAGE                = int(os.getenv("TREND_LEVERAGE", "5"))
ADX_CHOP_FLOOR                = float(os.getenv("ADX_CHOP_FLOOR", "15.0"))
HURST_TREND_FLOOR             = float(os.getenv("HURST_TREND_FLOOR", "0.52"))
PER_SYMBOL_DD_LIMIT           = float(os.getenv("PER_SYMBOL_DD_LIMIT", "-2.0"))
PROGRESSIVE_SCALING_THRESHOLD = 0.05  # Reduce size if DD > 5%

# Consecutive SL guard
CONSECUTIVE_SL_LIMIT          = int(os.getenv("CONSECUTIVE_SL_LIMIT", "10" if IS_REPLAY else "3"))
CONSECUTIVE_SL_COOLDOWN_MINS  = int(os.getenv("CONSECUTIVE_SL_COOLDOWN_MINS", "120"))
CONSECUTIVE_SL_HALT_COUNT     = 3
CONSECUTIVE_SL_HALT_HOURS     = 24

# Volatility / Sudden-move guards (very permissive in replay so they never trigger)
SUDDEN_MOVE_THRESHOLD_PERCENT = float(os.getenv("SUDDEN_MOVE_THRESHOLD_PERCENT", "1000.0" if IS_REPLAY else "3.0"))
ATR_SPIKE_MULTIPLIER          = float(os.getenv("ATR_SPIKE_MULTIPLIER", "10.0"  if IS_REPLAY else "2.0"))
VOLATILITY_90PCT_CONSECUTIVE  = int(os.getenv("VOLATILITY_90PCT_CONSECUTIVE", "1000" if IS_REPLAY else "3"))
MAX_PRICE_CHANGE_PCT          = float(os.getenv("MAX_PRICE_CHANGE_PCT", "1000.0" if IS_REPLAY else "3.0"))

# Cooldowns & limits
POST_LOSS_COOLDOWN_MINUTES    = int(os.getenv("POST_LOSS_COOLDOWN_MINUTES", "0" if IS_REPLAY else "120"))
POST_TP_COOLDOWN_MINUTES      = int(os.getenv("POST_TP_COOLDOWN_MINUTES", "120"))
EARLY_BE_ATR_MULT             = float(os.getenv("EARLY_BE_ATR_MULT", "1.0"))
BREAKEVEN_FEE_BUFFER_PCT      = float(os.getenv("BREAKEVEN_FEE_BUFFER_PCT", "0.001"))
GLOBAL_HALT_COOLDOWN_HOURS    = int(os.getenv("GLOBAL_HALT_COOLDOWN_HOURS", "24"))
FAST_SL_TIME_MINS             = int(os.getenv("FAST_SL_TIME_MINS", "5"))
MAX_POSITIONS_PER_SYMBOL      = int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "4"))
GLOBAL_MAX_DISTINCT_COINS     = int(os.getenv("GLOBAL_MAX_DISTINCT_COINS", "8"))
MAX_PORTFOLIO_MARGIN_PCT      = float(os.getenv("MAX_PORTFOLIO_MARGIN_PCT", "0.80"))
ALLOW_SAME_COIN_SCALE_IN      = os.getenv("ALLOW_SAME_COIN_SCALE_IN", "true").lower() in ("true", "1")
RISK_VOLATILITY_CAP           = float(os.getenv("RISK_VOLATILITY_CAP", "0.5"))
RISK_PHASE_MATRIX             = {"EARLY": 1.2, "MATURE": 1.0, "EXHAUSTED": 0.5, "CHOP": 0.3, "UNKNOWN": 0.5}

# Funding gate
MAX_ALLOWED_FUNDING_RATE      = float(os.getenv("MAX_ALLOWED_FUNDING_RATE", "0.0001"))  # 0.01%
EXTREME_FUNDING_THRESHOLD     = 0.001   # 0.1%
FUNDING_EXIT_WINDOW_MINS      = 15
FAST_SL_THRESHOLD_MINS        = 5

# News gate
PRE_NEWS_MINUTES             = int(os.getenv("PRE_NEWS_MINUTES", "60"))
POST_NEWS_MINUTES            = int(os.getenv("POST_NEWS_MINUTES", "120"))
NEWS_IMPACT_FILTER           = ["High", "Medium"]
KILL_SWITCH_FILE              = "STOP_TRADING"

# Correlation guards
MAX_CORRELATED_EXPOSURE             = float(os.getenv("MAX_CORRELATED_EXPOSURE", "2.0"))
MAX_DIRECTIONAL_EXPOSURE_PER_SYMBOL = float(os.getenv("MAX_DIRECTIONAL_EXPOSURE_PER_SYMBOL", "1.5"))
ANTI_PYRAMID_ATR_BUFFER             = float(os.getenv("ANTI_PYRAMID_ATR_BUFFER", "1.5"))

# ---------------------------------------------------------------------------
# ADX / Regime Thresholds
# ---------------------------------------------------------------------------
ADX_WEAK_THRESHOLD            = int(os.getenv("ADX_WEAK_THRESHOLD", "10"))
ADX_STRONG_THRESHOLD          = int(os.getenv("ADX_STRONG_THRESHOLD", "25"))
ADX_EXHAUSTION_THRESHOLD      = int(os.getenv("ADX_EXHAUSTION_THRESHOLD", "45"))
ADX_TREND_THRESHOLD           = 20.0
ADX_RANGING_MULT              = 0.8
ADX_SMA_FLOOR                 = 15.0
EMA_DIV_EXTREME               = float(os.getenv("EMA_DIV_EXTREME", "3.0"))

# Market readiness
SLOPE_TREND_THRESHOLD         = 0.1
SLOPE_RANGING_THRESHOLD       = 0.05
SLOPE_CHAOTIC_THRESHOLD       = 0.2
ATR_CHAOTIC_THRESHOLD         = 3.0
MIN_TRENDING_CANDLES          = 4
MIN_ADX_READY                 = 15.0

# ---------------------------------------------------------------------------
# Institutional Alpha Parameters
# ---------------------------------------------------------------------------
TREND_ACCELERATION_THRESHOLD  = 1.2    # ADX / ADX_SMA ratio to confirm trend entry
COUNTER_TREND_SIZE_REDUCTION  = 0.25   # Size multiplier for counter-trend entries

LATE_TREND_ADX_MIN            = 30.0
LATE_TREND_ACCEL_MIN          = 1.1
LATE_TREND_SIZE_REDUCTION     = 0.5

# ---------------------------------------------------------------------------
# V6 Macro Gate (4H MTF Filter)
# ---------------------------------------------------------------------------
HTF_TIMEFRAME    = os.getenv("HTF_TIMEFRAME", "4h")
HTF_EMA_PERIOD   = int(os.getenv("HTF_EMA_PERIOD", "200"))
HTF_ENABLED      = os.getenv("HTF_ENABLED", "True").lower() in ("true", "1")
HTF_MIN_CANDLES  = 1000   # Matches OHLCV_LIMIT for a stable 4H EMA200

# ---------------------------------------------------------------------------
# Regime Parameters (V6)
# ---------------------------------------------------------------------------
REGIME_SLOPE_THRESHOLD = 0.0005
TREND_TP_MULTIPLE      = 2.5
CHOP_TP_MULTIPLE       = 1.0

# ---------------------------------------------------------------------------
# Capital Protection & Throttling (V6.2)
# ---------------------------------------------------------------------------
MIN_AI_CONVICTION_SCORE   = 50.0
MIN_NN_CONFIDENCE         = float(os.getenv("MIN_NN_CONFIDENCE", "0.34"))
MAX_TRADES_PER_24H        = 3
MAX_CONCURRENT_TRADES     = int(os.getenv("MAX_CONCURRENT_TRADES", "8"))
RISK_PER_TRADE_PCT        = float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))  # Single source of truth (0.5% equity risk per trade)
# P3-PATCH: Raised from 5.0 → 55.0. At 5.0 the gate was functionally dead (any
# conviction > 5/100 passed). 55.0 creates a meaningful second-tier filter:
# Boardroom must score ≥50 to signal (MIN_AI_CONVICTION_SCORE), then Berserker
# must score ≥55 to open — adding a 5-point buffer above the signal floor.
BERSERKER_VETO_THRESHOLD  = float(os.getenv("BERSERKER_VETO_THRESHOLD", "55.0"))  # AI conviction floor for Berserker entry
MAX_DD_PER_TRADE_PCT      = 0.5

# Adaptive Risk Maps
ADAPTIVE_SL_MAP = {"normal": 2.0, "sniper": 1.5, "aggressive": 2.5}
ADAPTIVE_TP_MAP = {"swing": 3.0, "scalp": 1.5, "trend": 4.0}
ADAPTIVE_TRAIL_MAP = {"normal": 1.5, "tight": 1.0, "loose": 3.0}

# Elite Displacement Strategy (V8.5 / V36.5 Optimized Payoff Geometry)
LONG_SL_ATR_MULT = float(os.getenv("LONG_SL_ATR_MULT", "1.5"))
LONG_TP_ATR_MULT = float(os.getenv("LONG_TP_ATR_MULT", "4.5"))
LONG_TRAIL_ACTIVATION_ATR = float(os.getenv("LONG_TRAIL_ACTIVATION_ATR", "2.0"))
LONG_TRAIL_DIST_ATR = float(os.getenv("LONG_TRAIL_DIST_ATR", "2.0"))

SHORT_SL_ATR_MULT = float(os.getenv("SHORT_SL_ATR_MULT", "1.5"))
SHORT_TP_ATR_MULT = float(os.getenv("SHORT_TP_ATR_MULT", "4.5"))
SHORT_TRAIL_ACTIVATION_ATR = float(os.getenv("SHORT_TRAIL_ACTIVATION_ATR", "2.0"))
SHORT_TRAIL_DIST_ATR = float(os.getenv("SHORT_TRAIL_DIST_ATR", "2.0"))

# Dynamic Regime-Specific Hyperparameter Profiles (V36.0)
REGIME_PROFILES = {
    "BULLISH": {
        "min_conviction": float(os.getenv("BULL_MIN_CONVICTION", "55.0")),
        "tp_atr_mult": float(os.getenv("BULL_TP_ATR_MULT", "4.5")),
        "sl_atr_mult": float(os.getenv("BULL_SL_ATR_MULT", "1.5")),
        "trail_activation_atr": float(os.getenv("BULL_TRAIL_ATR", "2.0")),
        "leverage": int(os.getenv("BULL_LEVERAGE", "5")),
        "size_multiplier": 1.0,
    },
    "BEARISH": {
        "min_conviction": float(os.getenv("BEAR_MIN_CONVICTION", "55.0")),  # Symmetrical Alpha: 55.0 matches Bull threshold
        "tp_atr_mult": float(os.getenv("BEAR_TP_ATR_MULT", "4.5")),         # Trend runner capture on breakdowns
        "sl_atr_mult": float(os.getenv("BEAR_SL_ATR_MULT", "1.8")),         # Tight SL against violent short squeezes
        "trail_activation_atr": float(os.getenv("BEAR_TRAIL_ATR", "2.0")),
        "leverage": int(os.getenv("BEAR_LEVERAGE", "5")),                    # Symmetrical 5x leverage
        "size_multiplier": 1.0,
    },
    "CHOP": {
        "min_conviction": float(os.getenv("CHOP_MIN_CONVICTION", "70.0")),  # Very high conviction required in chop
        "tp_atr_mult": float(os.getenv("CHOP_TP_ATR_MULT", "3.0")),         # Scalp exits
        "sl_atr_mult": float(os.getenv("CHOP_SL_ATR_MULT", "1.5")),         # Tight SL
        "trail_activation_atr": float(os.getenv("CHOP_TRAIL_ATR", "2.0")),
        "leverage": int(os.getenv("CHOP_LEVERAGE", "3")),
        "size_multiplier": 0.5,                                             # Half size
    }
}

def get_regime_profile(regime_or_trend: str = "BULLISH") -> dict:
    key = str(regime_or_trend).upper()
    if "CHOP" in key or "RANG" in key or "CHAOTIC" in key or "NEUTRAL" in key:
        return REGIME_PROFILES["CHOP"]
    elif "BEAR" in key or "SHORT" in key:
        return REGIME_PROFILES["BEARISH"]
    return REGIME_PROFILES["BULLISH"]

# Partial Take Profit (configured via env — see lines 70-71)

# ---------------------------------------------------------------------------
# LLM / AI (Multi-Tier: Google Gemini Primary + Groq Failover + Multi-Model Pools)
# ---------------------------------------------------------------------------
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_MODEL_POOL  = [
    m.strip() for m in os.getenv(
        "GEMINI_MODEL_POOL",
        "gemini-flash-lite-latest,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-3.6-flash,gemini-3.1-flash-lite-preview"
    ).split(",") if m.strip()
]
GEMINI_URL         = os.getenv("GEMINI_URL", "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_URL           = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL         = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_MODEL_POOL    = [
    m.strip() for m in os.getenv(
        "GROQ_MODEL_POOL",
        "openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b"
    ).split(",") if m.strip()
]
LLM_TIMEOUT        = 120
LLM_ATTEMPT_TIMEOUT = int(os.getenv("LLM_ATTEMPT_TIMEOUT", "180"))

# ---------------------------------------------------------------------------
# Vol Gate Multipliers (V11.0 — extracted from hardcoded bot.py values)
# ---------------------------------------------------------------------------
SNIPER_VOL_GATE_MULT    = float(os.getenv("SNIPER_VOL_GATE_MULT", "1.5"))
# ---------------------------------------------------------------------------
# Quantitative Microstructure & Reservation Price Parameters
# ---------------------------------------------------------------------------
USE_DETERMINISTIC_MICROSTRUCTURE = os.getenv("USE_DETERMINISTIC_MICROSTRUCTURE", "true").lower() in ("true", "1")
AS_GAMMA_RISK_AVERSION           = float(os.getenv("AS_GAMMA_RISK_AVERSION", "1.5"))
# DIAGNOSTIC BOOT (V11.0 — uses logger, not print, to avoid stdout pollution)
CACHE_TTL_SECONDS  = int(os.getenv("CACHE_TTL_SECONDS", "300"))
_log.info(f"[SYSTEM] Config Loaded: Balance=${INITIAL_BALANCE}, Risk={RISK_PER_TRADE_PCT}%")

