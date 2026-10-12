import pandas as pd
import os
import logging
from typing import Dict, Optional, Tuple
import config

log = logging.getLogger(__name__)

class HistoricalBuffer:
    def __init__(self, symbol: str, timeframe: str, filepath: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = self._load_csv(filepath)
        self.cursor = 0 # Current index in replay
        self.max_idx = len(self.df) - 1

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Replay data not found at {filepath}")
        
        df = pd.read_csv(filepath)
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        
        # Robust Timestamp Detection
        t_col = 'timestamp' if 'timestamp' in df.columns else 'time' if 'time' in df.columns else None
        
        if t_col:
            try:
                # First try numeric milliseconds
                first_val = float(df[t_col].iloc[0])
                if first_val > 1e12: # Milliseconds
                    df[t_col] = pd.to_datetime(df[t_col], unit='ms', utc=True)
                else: # Seconds
                    df[t_col] = pd.to_datetime(df[t_col], unit='s', utc=True)
            except:
                # ISO string fallback
                df[t_col] = pd.to_datetime(df[t_col], utc=True)
            
            df.set_index(t_col, inplace=True)
        else:
            log.warning(f"No timestamp column found in {filepath}. Using integer index.")
            
        df = df.sort_index()
        
        # Apply date filtering for replay if in config
        log.debug(f"HistoricalBuffer load: {filepath} | IS_REPLAY={config.IS_REPLAY}")
        if config.IS_REPLAY:
            start = pd.to_datetime(config.REPLAY_START_DATE, utc=True)
            end = pd.to_datetime(config.REPLAY_END_DATE, utc=True)
            
            # We need to include warmup data before start, so we fetch OHLCV_LIMIT candles before start
            # But the simplest way is to just filter after loading and let the cursor handle it.
            # However, HistoricalBuffer.get_slice uses cursor index.
            # So we should filter the DF to [start - OHLCV_LIMIT candles, end]
            
            mask = df.index <= end
            df = df[mask]
            
            start_idx_in_full = df.index.get_indexer([start], method='bfill')[0]
            if start_idx_in_full != -1:
                # Include warmup
                actual_start_idx = max(0, start_idx_in_full - config.OHLCV_LIMIT)
                df = df.iloc[actual_start_idx:]
                
        return df

    def get_slice(self, limit: int) -> pd.DataFrame:
        """Returns a window of data ending at the current cursor."""
        start_idx = max(0, self.cursor - limit + 1)
        return self.df.iloc[start_idx : self.cursor + 1]

    def get_slice_up_to_ts(self, timestamp: pd.Timestamp, limit: int) -> pd.DataFrame:
        """Returns data from this timeframe up to (or before) the given timestamp."""
        # Find the last index where timestamp <= given timestamp
        mask = self.df.index <= timestamp
        filtered = self.df[mask]
        return filtered.tail(limit)

    def advance(self) -> bool:
        """Move cursor forward. Returns False if end of data."""
        if self.cursor < self.max_idx:
            self.cursor += 1
            return True
        return False

    def get_current_timestamp(self) -> pd.Timestamp:
        return self.df.index[self.cursor]

    def get_current_price(self) -> float:
        return float(self.df.iloc[self.cursor]['close'])

class ReplayManager:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Key: (symbol, timeframe)
        self.buffers: Dict[Tuple[str, str], HistoricalBuffer] = {}
        self.is_active = False

    def load_symbol(self, symbol: str, timeframe: str):
        """Loads a specific timeframe for a symbol."""
        clean_sym = symbol.replace("/", "")
        filename = f"{clean_sym}_{timeframe}.csv"
        path = os.path.join(self.data_dir, filename)
        
        log.info(f"Loading replay buffer for {symbol} {timeframe} from {path}")
        self.buffers[(symbol, timeframe)] = HistoricalBuffer(symbol, timeframe, path)
        self.is_active = True

    def get_window(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        key = (symbol, timeframe)
        from config import TIMEFRAME as PRIMARY_TF
        primary_key = (symbol, PRIMARY_TF)
        
        if primary_key not in self.buffers:
            try:
                self.load_symbol(symbol, PRIMARY_TF)
            except:
                raise KeyError(f"No replay buffer for {symbol} {PRIMARY_TF}")
        
        # Primary timeframe (usually 1h) is used to drive the clock
        current_ts = self.buffers[primary_key].get_current_timestamp()

        if timeframe == PRIMARY_TF:
            return self.buffers[primary_key].get_slice(limit)
        else:
            # Phase 17: Resample 1h -> 4h for Macro Gate
            # We need enough 1h data to produce 'limit' HTF candles.
            # Fetch all 1h data up to the current clock
            h1_full = self.buffers[primary_key].df
            h1_window = h1_full[h1_full.index <= current_ts]
            
            # Resample 1h to HTF (e.g. 4h)
            # label='left' means the timestamp is the start of the candle (CCXT standard)
            # closed='left' means the start is inclusive, end is exclusive
            htf_df = h1_window.resample(timeframe, label='left', closed='left').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            })
            
            # CRITICAL: Drop the last candle if it is still "forming"
            # A 4h candle starting at T is finished only if the next 4h candle (T + 4h) would have started
            # Or more simply, if current_ts >= T + 3h (for 1h steps)
            # Actually, in 1h steps, at 03:00 we have the 4th candle of the [00:00-04:00) block.
            # But the 04:00 1h candle is the one that signals the 00:00 4h candle is FIXED.
            
            # To be safe and avoid look-ahead, we only keep candles where the index + timeframe <= current_ts
            # Wait, that's too restrictive. If current_ts is 03:00, the 00:00 4h candle is "closed" 
            # in the sense that its High/Low/Close for that 4h block are finalized once we move to 04:00.
            
            # Logic: If current_ts is the Nth candle of a period, the period is only "closed" if we are at or past the END of it.
            # For 4h, if current_ts is 00:00, 01:00, 02:00, 03:00 -> 00:00 candle is forming.
            # At 04:00 -> 00:00 candle is CLOSED.
            
            # Calculate the expected duration in minutes
            import re
            match = re.match(r"(\d+)([hdm])", timeframe)
            if match:
                num, unit = int(match.group(1)), match.group(2)
                offset = pd.Timedelta(hours=num) if unit == 'h' else pd.Timedelta(days=num) if unit == 'd' else pd.Timedelta(minutes=num)
                
                # Keep only candles that are fully formed relative to current_ts
                htf_df = htf_df[htf_df.index + offset <= current_ts]
            
            return htf_df.tail(limit)

    def advance_all(self, primary_tf: str):
        """Advances only the primary timeframe buffers. Returns {symbol: success}."""
        res = {}
        for (sym, tf), buf in self.buffers.items():
            if tf == primary_tf:
                res[sym] = buf.advance()
        return res

    def get_price(self, symbol: str, primary_tf: str) -> float:
        return self.buffers[(symbol, primary_tf)].get_current_price()
