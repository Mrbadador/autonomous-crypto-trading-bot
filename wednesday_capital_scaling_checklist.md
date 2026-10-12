# Wednesday Post-FOMC Capital Scaling Scorecard & Audit Checklist

**Evaluation Target**: Wednesday, September 16, 2026 (Post-FOMC Close / ~21:00 UTC)  
**Baseline Recorded**: Sunday, September 13, 2026 (01:00 UTC)  
**Objective**: Empirically evaluate fleet robustness across 3 distinct market regimes before authorizing additional capital injection.

---

## 1. Baseline Fleet Metrics (Sept 13, 01:00 UTC)

* **Bitget UTA Balance**: `$60.5319 USDT` (Baseline before reload: `$58.7700`, +3.00% net)
* **Margin Utilization**: `$13.23 USDT` (~21.8% of portfolio)
* **Free Cash Reserve**: `$47.30 USDT` (~78.2% cash buffer)
* **Active Open Positions**:
  * `ADA/USDT` SHORT: Qty 179.0 @ $0.2060 (SL: $0.2122 | TP: $0.1957 | HardSL: $0.2163)
  * `AVAX/USDT` LONG: Qty 3.9 @ $7.5950 (SL: $7.4431 | TP: $7.8988 | HardSL: $7.3671)
* **Closed Trades in Current Stack Session**:
  * `BTC/USDT` SHORT: Exit at $77,099.70 | PnL: **+$0.0123** (Dynamic Trailing Stop)
  * `LINK/USDT` SHORT: Exit at $11.5160 | PnL: **+$0.0210** (Dynamic Trailing Stop)
* **Infrastructure State**: 
  * SQLite Lock Contention: 0 errors / 9.5h continuous runtime
  * Reconciler Desyncs: 0
  * Hourly Sentinel SLA Compliance: 100%

---

## 2. The 4-Phase Evaluation Runway

### Phase 1: Weekend Low-Volume & CME Open (Sunday, Sept 13)
- [ ] **Hurst Exponent & ADX Throttle**: Did the bot avoid over-trading during low-volume weekend chop?
- [ ] **Hawkes Process Vetoes**: Were phantom book liquidity sweeps and spoofed wicks filtered out?
- [ ] **Dynamic Leverage**: Did leverage compress to 3x if low-volatility chop was detected?

### Phase 2: Weekly Open Directional Alignment (Monday, Sept 14)
- [ ] **Macro Trend Alignment**: Did `btc_macro_trend` accurately align with early Monday institutional flow?
- [ ] **Anti-Fighting Rule**: Were counter-trend altcoin entries blocked when BTC established weekly momentum?

### Phase 3: Wednesday FOMC Event Shield (Wednesday, Sept 16, 18:00 UTC)
- [ ] **T-60m Pre-Event Lockout (17:00 UTC)**: Did `economic_calendar.py` actively suppress new entries?
- [ ] **T-0 Volatility Absorption (18:00 UTC)**: Were existing positions safely shielded by native Bitget hard stops without slippage blowouts?
- [ ] **T+120m Post-Event Freeze (until 20:00 UTC)**: Did the bot stay flat during post-announcement whipsaws before resuming sniper entries?

### Phase 4: Statistical Edge Scorecard (Wednesday Post-Market Audit)
- [ ] **Sample Size ($N$)**: Total closed trades accumulated ($N \ge 20\text{--}30$)
- [ ] **Profit Factor**: Gross Profits / Gross Losses $> 1.30$
- [ ] **Net Expectancy**: $E = (W \times \text{Avg Win}) - (L \times \text{Avg Loss}) > \text{Fees}$
- [ ] **Max Drawdown**: Kept strictly below $10.0\%$ of peak equity
- [ ] **Trailing Stop Efficiency**: Did dynamic trailing stops preserve green PnL on trend reversals?

---

## 3. Capital Scaling Decision Matrix

| Metric Outcome | Quantitative Verdict | Action |
| :--- | :--- | :--- |
| **All 4 Phases Satisfied & PF > 1.30** | **GREEN LIGHT** | **Scale account to $150–$250 USDT.** Unlocks multi-asset diversification (4–5 concurrent positions) without lot-size quantization friction. |
| **Capital Preserved, but $N < 15$ Trades** | **YELLOW LIGHT** | **Hold capital.** System is safe, but statistical sample is too small. Continue running on $60 wallet until $N \ge 30$. |
| **Max Drawdown > 10% or Negative Expectancy** | **RED LIGHT** | **VETO.** Do not add capital. Review post-mortem logs to recalibrate entry confidence or regime thresholds. |

---

## 4. One-Click Audit Script for Wednesday

To run this audit automatically on Wednesday night, execute:
```powershell
.venv\Scripts\python.exe -c "
import sqlite3, pandas as pd
conn = sqlite3.connect('file:trading_data.db?mode=ro', uri=True, timeout=5.0)
df = pd.read_sql_query(\"SELECT symbol, side, pnl, exit_price, timestamp FROM trades WHERE timestamp >= '2026-09-12T10:30:00' AND pnl IS NOT NULL\", conn)
conn.close()
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] < 0]
gross_profit = wins['pnl'].sum()
gross_loss = abs(losses['pnl'].sum())
pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)
wr = len(wins) / len(df) if len(df) > 0 else 0.0
print(f'=== WEDNESDAY SCORECARD ===')
print(f'Total Trades (N): {len(df)}')
print(f'Win Rate: {wr:.1%} ({len(wins)}W / {len(losses)}L)')
print(f'Gross Profit: +${gross_profit:.4f} | Gross Loss: -${gross_loss:.4f}')
print(f'Profit Factor: {pf:.2f}')
print(f'Net PnL: ${df[\"pnl\"].sum():+.4f}')
"
```
