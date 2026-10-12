import re
import pandas as pd
from datetime import datetime, UTC
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console, Group
from rich import box
from rich.text import Text
from rich.progress import ProgressBar
from rich.columns import Columns
from rich.align import Align
import config

console = Console()

# ---------------------------------------------------------------------------
# ANSI Colors (kept for compatibility)
# ---------------------------------------------------------------------------
RESET   = "\033[0m"
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"

def strip_ansi(text: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', text)

def generate_sparkline(chart_data: list, side: str = "long") -> Text:
    """Generates a position-aware sparkline. Green = going your way, red = against you."""
    if not chart_data:
        return Text("")
    closes = [float(c.get('close', c.get('low', 0))) for c in chart_data]
    if len(closes) < 2:
        return Text("")
    
    ticks = [' ', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    text = Text()
    
    # Determine slope direction relative to position side
    # For longs: rising = good (green), falling = bad (red)
    # For shorts: rising = bad (red), falling = good (green)
    for i in range(1, len(closes)):
        prev, curr = closes[i-1], closes[i]
        slope = curr - prev
        
        # Normalize to tick index based on relative magnitude within the range
        all_vals = closes[:i+1]
        min_val, max_val = min(all_vals), max(all_vals)
        if max_val == min_val:
            idx = 3
        else:
            idx = int(((curr - min_val) / (max_val - min_val)) * 7)
            idx = max(0, min(7, idx))
        
        # Color by whether this move helps or hurts the position
        if side == "long":
            color = "green" if slope >= 0 else "red"
        else:
            color = "green" if slope <= 0 else "red"
        
        text.append(ticks[idx], style=color)
    
    return text

def format_percent(value: float, color=True) -> str:
    sign = "+" if value >= 0 else ""
    text = f"{sign}{value:.2f}%"
    if color:
        if value > 0:
            return f"[green]{text}[/green]"
        elif value < 0:
            return f"[red]{text}[/red]"
    return text

def format_currency(value: float) -> str:
    return f"${value:,.2f}"

def get_position_badge(pos, current_price: float) -> tuple:
    """Returns (badge_text, badge_style) for position state."""
    # Calculate true breakeven with side awareness
    fee_adjustment = 1 + config.FEE_RATE * 2
    if pos.side == "long":
        true_be = pos.entry_price * fee_adjustment
    else:
        true_be = pos.entry_price * (2 - fee_adjustment)  # Subtract fees for shorts
    
    # Distance to stop loss as percentage
    sl_dist_pct = abs((current_price - pos.sl) / pos.entry_price * 100) if pos.entry_price else 0
    
    if pos.unr >= 0 and current_price >= true_be:
        return ("[BE+]", "bold green")
    elif sl_dist_pct < 1.0:
        return ("[RISK]", "bold red")
    elif pos.unr >= 0:
        return ("[PROFIT]", "green")
    else:
        return ("[OPEN]", "dim")

def compute_trade_stats(post_trade_notes: list) -> dict:
    """Compute session statistics from trade history."""
    if not post_trade_notes:
        return {}
    
    pnls = [t.get('pnl', 0) for t in post_trade_notes]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    
    total_trades = len(pnls)
    win_count = len(wins)
    loss_count = len(losses)
    
    stats = {
        'total_trades': total_trades,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': (win_count / total_trades * 100) if total_trades > 0 else 0,
        'avg_win': sum(wins) / len(wins) if wins else 0,
        'avg_loss': sum(losses) / len(losses) if losses else 0,
        'best_trade': max(pnls) if pnls else 0,
        'worst_trade': min(pnls) if pnls else 0,
        'profit_factor': (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf') if wins else 0,
        'total_pnl': sum(pnls)
    }
    return stats

def build_dashboard_layout(state, results, em) -> Layout:
    """Constructs an institutional-grade trading desk UI."""
    layout = Layout()

    # Define grid: header, alert ticker, body, footer
    layout.split(
        Layout(name="header", size=4),
        Layout(name="alert_ticker", size=1),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=2)
    )
    layout["left"].split(
        Layout(name="alpha_matrix", ratio=1),
        Layout(name="market_action", ratio=1),
        Layout(name="system_health", ratio=1)
    )
    layout["right"].split(
        Layout(name="positions", ratio=2),
        Layout(name="trade_stats", ratio=1),
        Layout(name="reasoning", ratio=1)
    )

    # 1. Header: Global Performance & Ticker
    bal = em.balance
    perf = (bal.total - config.INITIAL_BALANCE) / config.INITIAL_BALANCE * 100
    perf_color = "bold green" if perf >= 0 else "bold red"
    
    # Macro Trend Styling (Fix for MIXED crash)
    trend = state.global_btc_trend.lower()
    macro_border = "green" if "bullish" in trend else "red" if "bearish" in trend else "yellow"
    
    # Ticker data — minimal, scannable
    ticker_items = []
    for res in results:
        if not res: continue
        sym = res["symbol"].replace("/USDT", "")
        price = res["price"]
        slope = res["slope"]
        s_col = "green" if slope > 0 else "red"
        ticker_items.append(f"[cyan]{sym}[/cyan] {price:,.2f} [{s_col}]{slope:+.3f}%[/{s_col}]")
    
    header_content = Table.grid(expand=True)
    header_content.add_column(ratio=1)
    header_content.add_column(ratio=2)
    
    # Drawdown gauge
    current_dd = state.get_drawdown()
    max_dd = state.max_drawdown
    dd_limit = config.GLOBAL_MAX_DD_LIMIT
    dd_color = "green" if current_dd < 5 else "yellow" if current_dd < 10 else "red"
    dd_bar_width = 10
    dd_filled = int(min(current_dd / dd_limit, 1.0) * dd_bar_width) if dd_limit > 0 else 0
    dd_bar = "█" * dd_filled + "░" * (dd_bar_width - dd_filled)
    
    perf_text = Text.assemble(
        (" PORTFOLIO ", "bold black on cyan"),
        (f"  {format_currency(bal.total)} ", "bold white"),
        (f"({perf:+.2f}%)", perf_color),
        ("  STREAK: ", "white"), (f"{state.global_streak}", "bold green" if state.global_streak > 0 else "bold red" if state.global_streak < 0 else "dim"),
        ("  DD: ", "white"), (f"{current_dd:.1f}%", dd_color),
        ("  MAX: ", "dim"), (f"{max_dd:.1f}%", "dim red" if max_dd > 10 else "dim"),
        (f" [{dd_bar}]", dd_color)
    )

    ticker_text = Text.from_markup("  |  ".join(ticker_items))
    
    header_content.add_row(
        Panel(perf_text, border_style="cyan", padding=(0,1)),
        Panel(Align.center(ticker_text), border_style=macro_border, title=f"[dim]Market ({state.global_btc_trend})[/dim]")
    )
    layout["header"].update(header_content)

    # 2. Alert Ticker — dedicated line for critical alerts
    alert_items = []
    now = datetime.now(UTC)
    
    # Funding warnings
    funding_alert = False
    for res in results:
        if not res: continue
        rate = res.get("rate", 0)
        min_f = res.get("min_to_fund", 999)
        for p in em.positions:
            if p.symbol == res["symbol"] and p.unr > 0 and abs(rate) > 0.0001 and min_f < 60:
                funding_alert = True
                break
    if funding_alert:
        alert_items.append(("⚠ FUNDING SOON", "bold yellow"))
    
    # News freeze
    upcoming = [n for n in state.upcoming_news if n[0] > now]
    if upcoming:
        upcoming.sort(key=lambda x: x[0])
        n_time, n_impact, n_title = upcoming[0]
        mins_left = int((n_time - now).total_seconds() / 60)
        if mins_left < config.PRE_NEWS_MINUTES:
            alert_items.append((f"📡 NEWS FREEZE: {n_title} ({mins_left}m)", "bold orange1"))
        else:
            alert_items.append((f"📡 Next: {n_title} ({mins_left}m)", "dim"))
    
    # Halt status
    if state.is_halted:
        alert_items.append(("🛑 TRADING HALTED", "bold red"))
    elif state.is_observe_only:
        alert_items.append(("👁 OBSERVE ONLY", "bold yellow"))
    
    if alert_items:
        alert_text = Text.assemble(*[(f" {text} ", style) for text, style in alert_items])
        layout["alert_ticker"].update(Panel(alert_text, border_style="none", padding=(0,1), style="on grey11"))
    else:
        layout["alert_ticker"].update(Text(""))

    # 3. Alpha Matrix (Detailed Context)
    alpha_table = Table(show_header=True, box=box.SIMPLE, expand=True)
    alpha_table.add_column("Symbol", style="cyan")
    alpha_table.add_column("Phase", justify="center")
    alpha_table.add_column("ADX", justify="right")
    alpha_table.add_column("ADX_SMA", justify="right")
    alpha_table.add_column("ATR%", justify="right")
    alpha_table.add_column("Slope", justify="right")
    
    for res in results:
        if not res: continue
        sym = res["symbol"]
        phase = res.get("phase", "UNKNOWN")
        adx = res.get("adx", 0)
        atr_p = res.get("atr_pct", 0)
        slope = res.get("slope", 0)
        
        p_color = "green" if phase == "EARLY" else "yellow" if phase == "MATURE" else "red"
        adx_col = "bold green" if adx > 30 else "dim" if adx < 20 else "white"
        
        alpha_table.add_row(
            sym,
            f"[{p_color}]{phase}[/{p_color}]",
            f"[{adx_col}]{adx:.1f}[/{adx_col}]",
            f"{res.get('adx_sma', 0):.1f}",
            f"{atr_p * 100:.2f}%",
            f"{slope:+.4f}"
        )
    layout["alpha_matrix"].update(
        Panel(alpha_table, title="[white]Alpha Matrix[/white]", border_style="none", style="on grey7")
    )

    # 4. Market Action (Sparklines)
    action_group = []
    for res in results:
        if not res: continue
        sym = res["symbol"]
        # Find if we have a position to determine sparkline coloring
        pos_side = "long"
        for p in em.positions:
            if p.symbol == sym:
                pos_side = p.side
                break
        spark = generate_sparkline(res.get("chart_data", []), side=pos_side)
        action_group.append(Text.assemble((f"{sym: <10}", "cyan"), spark))
    
    layout["market_action"].update(
        Panel(Group(*action_group), title="[white]Price Action[/white]", border_style="none", style="on grey7")
    )

    # 5. System Health
    health_table = Table.grid(expand=True, padding=(0,1))
    health_table.add_column(style="dim")
    health_table.add_column(justify="right")
    
    # Find active risk mode from results
    active_mode = "NORMAL"
    for res in results:
        if res and "risk_mode" in res:
            active_mode = res["risk_mode"]
            break
            
    mode_color = "bold green" if active_mode == "AGGRESSIVE" else "bold cyan" if active_mode == "SNIPER" else "bold red" if active_mode == "CHOP" else "white"
    
    health_table.add_row("Engine Mode:", f"[{mode_color}]{active_mode}[/{mode_color}]")
    health_table.add_row("Uptime Cycles:", str(state.cycle_count))
    health_table.add_row("LLM Failures:", f"[red]{state.llm_failures}[/red]" if state.llm_failures > 0 else "0")
    health_table.add_row("Kill Switch:", "[red]HALTED[/red]" if state.is_halted else "[green]READY[/green]")
    health_table.add_row("Observe Only:", "[yellow]ON[/yellow]" if state.is_observe_only else "OFF")
    
    is_frozen = any(abs((n[0] - now).total_seconds()) < config.PRE_NEWS_MINUTES * 60 for n in state.upcoming_news)
    news_status = "[bold red]FREEZE[/bold red]" if is_frozen else "[dim]Normal[/dim]"
    health_table.add_row("News Status:", news_status)
    
    layout["system_health"].update(
        Panel(health_table, title="[white]System Health[/white]", border_style="none", style="on grey7")
    )

    # 6. Position Command Center
    pos_table = Table(show_header=True, box=box.SIMPLE, expand=True)
    pos_table.add_column("Pos", style="bold")
    pos_table.add_column("Entry", justify="right")
    pos_table.add_column("Current", justify="right")
    pos_table.add_column("PnL", justify="right")
    pos_table.add_column("Progress", width=20)
    pos_table.add_column("SL / BE / TP", justify="center")
    
    # Margin radar calculation
    total_notional = 0.0
    for pos in em.positions:
        if pos.is_open:
            current_price = em.sim.last_prices.get(pos.symbol, pos.entry_price)
            total_notional += pos.qty * current_price
    
    margin_used = em.balance.used
    total_equity = em.balance.total
    leverage = total_notional / total_equity if total_equity > 0 else 0
    margin_pct = (margin_used / total_equity * 100) if total_equity > 0 else 0
    max_margin_pct = config.GLOBAL_MARGIN_LIMIT
    margin_color = "green" if margin_pct < max_margin_pct * 0.5 else "yellow" if margin_pct < max_margin_pct * 0.8 else "red"
    margin_bar_width = 10
    margin_filled = int(min(margin_pct / max_margin_pct, 1.0) * margin_bar_width) if max_margin_pct > 0 else 0
    margin_bar = "█" * margin_filled + "░" * (margin_bar_width - margin_filled)
    
    for pos in em.positions:
        pl_color = "green" if pos.unr >= 0 else "red"
        side_color = "green" if pos.side == "long" else "red"
        current_price = em.sim.last_prices.get(pos.symbol, pos.entry_price)
        
        # Calculate progress towards TP with side awareness
        dist_to_tp = abs(pos.tp - pos.entry_price)
        dist_current = abs(current_price - pos.entry_price)
        progress = min(1.0, dist_current / dist_to_tp) if dist_to_tp > 0 else 0
        
        # True breakeven with side awareness
        fee_adjustment = 1 + config.FEE_RATE * 2
        if pos.side == "long":
            true_be = pos.entry_price * fee_adjustment
        else:
            true_be = pos.entry_price * (2 - fee_adjustment)
        
        # Position badges
        badge, badge_style = get_position_badge(pos, current_price)
        
        pos_table.add_row(
            f"[{side_color}]{pos.symbol}[/]{badge}",
            f"{pos.entry_price:,.2f}",
            f"{current_price:,.2f}",
            f"[{pl_color}]{pos.unr:+.2f}[/{pl_color}]",
            ProgressBar(total=100, completed=progress*100, width=20, pulse=False),
            f"{pos.sl:.0f} | [{badge_style}]{true_be:.0f}[/] | {pos.tp:.0f}"
        )
    
    # Add margin radar at bottom of positions panel
    if em.positions:
        pos_table.add_row(
            "", "", "", "",
            f"[dim]Leverage: {leverage:.1f}x[/]",
            f"[{margin_color}]Margin: {format_currency(margin_used)} / {format_currency(total_equity)} [{margin_bar}][/{margin_color}]"
        )
    
    if not em.positions:
        layout["positions"].update(
            Panel(Align.center(Text("\n\nSCANNING MARKET FOR ALPHA...", style="dim italic")), 
                  title="[white]Command Center[/white]", border_style="cyan")
        )
    else:
        layout["positions"].update(
            Panel(pos_table, title="[white]Command Center[/white]", border_style="cyan")
        )

    # 7. Trade Stats (replacing simple history)
    stats = compute_trade_stats(state.post_trade_notes)
    if stats:
        stats_table = Table.grid(expand=True, padding=(0,1))
        stats_table.add_column(style="dim")
        stats_table.add_column(justify="right")
        stats_table.add_column(style="dim")
        stats_table.add_column(justify="right")
        
        wr_color = "green" if stats['win_rate'] > 50 else "yellow" if stats['win_rate'] > 40 else "red"
        pf_color = "green" if stats['profit_factor'] > 1.5 else "yellow" if stats['profit_factor'] > 1.0 else "red"
        
        stats_table.add_row(
            "Trades:", f"{stats['total_trades']}",
            "Win Rate:", f"[{wr_color}]{stats['win_rate']:.1f}%[/{wr_color}]"
        )
        stats_table.add_row(
            "Wins:", f"[green]{stats['win_count']}[/]",
            "Profit Factor:", f"[{pf_color}]{stats['profit_factor']:.2f}[/{pf_color}]"
        )
        stats_table.add_row(
            "Losses:", f"[red]{stats['loss_count']}[/]",
            "Total PnL:", f"{'[green]' if stats['total_pnl'] >= 0 else '[red]'}{stats['total_pnl']:+.2f}[/]"
        )
        stats_table.add_row(
            "Avg Win:", f"[green]{stats['avg_win']:+.2f}[/]",
            "Avg Loss:", f"[red]{stats['avg_loss']:+.2f}[/]"
        )
        stats_table.add_row(
            "Best:", f"[green]{stats['best_trade']:+.2f}[/]",
            "Worst:", f"[red]{stats['worst_trade']:+.2f}[/]"
        )
        
        # Last 3 trades compact
        recent = state.post_trade_notes[-3:]
        recent_texts = []
        for t in reversed(recent):
            ts = t['timestamp'].split('T')[1][:5] if 'T' in str(t['timestamp']) else str(t['timestamp'])[:5]
            pnl = t.get('pnl', 0)
            pnl_col = "green" if pnl > 0 else "red"
            recent_texts.append(f"[{pnl_col}]{ts} {t['symbol']} {pnl:+.2f}[/{pnl_col}]")
        
        recent_line = Text.from_markup("  |  ".join(recent_texts))
        
        layout["trade_stats"].update(
            Panel(Group(stats_table, Text(""), recent_line), 
                  title="[white]Session Stats[/white]", border_style="none", style="on grey7")
        )
    else:
        layout["trade_stats"].update(
            Panel(Text("No trades yet", style="dim"), 
                  title="[white]Session Stats[/white]", border_style="none", style="on grey7")
        )

    # 8. Reasoning Timeline
    reason_table = Table.grid(expand=True)
    for res in results:
        if not res: continue
        sym = res["symbol"]
        notes = res.get("ai_res", {}).get("notes", "...")
        action = res.get("final_action", "WAIT")
        a_col = "bold green" if "OPEN" in action else "bold red" if "EXIT" in action else "dim"
        
        reason_table.add_row(Text(f" {sym} » ", style="cyan") + Text(action, style=a_col))
        reason_table.add_row(Text(f" └─ {notes}\n", style="italic dim"))
        
    layout["reasoning"].update(
        Panel(reason_table, title="[white]AI Reasoning[/white]", border_style="none", style="on grey7")
    )

    # 9. Footer
    now_str = datetime.now(UTC).strftime("%H:%M:%S")
    footer_text = Text.assemble(
        (" STATUS: ", "white"), ("SCANNING" if not state.is_halted else "HALTED", "green" if not state.is_halted else "red"),
        ("  |  ", "dim"),
        ("MODEL: ", "white"), (config.OLLAMA_MODEL, "magenta"),
        ("  |  ", "dim"),
        ("CYCLES: ", "white"), (f"{state.cycle_count}", "white"),
        ("  |  ", "dim"),
        ("TIME: ", "white"), (f"{now_str} UTC", "dim")
    )
    layout["footer"].update(Panel(Align.center(footer_text), border_style="dim"))

    return layout

# Global live instance state hooks
_live_ctx = None

def setup_live():
    global _live_ctx
    if _live_ctx is None:
        _live_ctx = Live(auto_refresh=False, screen=False)
        _live_ctx.start()
    return _live_ctx

def update_live(layout):
    global _live_ctx
    if _live_ctx:
        console.clear()
        _live_ctx.update(layout, refresh=True)
    else:
        console.print(layout)

def stop_live():
    global _live_ctx
    if _live_ctx:
        _live_ctx.stop()
        _live_ctx = None
