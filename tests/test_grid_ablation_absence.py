"""
tests/test_grid_ablation_absence.py — F3 of the 2026-10-22 audit remediation.

Pins ``scripts/run_grid_ablation.py`` against the branch's governing defect class ("a label or
number in a report that claims more than the measurement supports"):

* an arm that closed no trades publishes the ABSENCE (``None`` + a reason) for win rate, profit
  factor, EV and the EV CI — ``n_trades == 0`` stays because it IS the measurement — instead of
  the measured-looking zeros the old empty-arm dict returned;
* ``bootstrap_ci([])`` returns ``(None, None, None)``, not a silent ``(0.0, 0.0, 0.0)``;
* a profit series with no losing trade (no denominator) publishes ``profit_factor is None``,
  never the fabricated stand-in ``99.0``; a mixed series still publishes the measured ratio;
* the Section-1 verdict cells are derived from the row's own measured numbers — no hard-coded
  figure may appear in any verdict cell — and row (c) states the shield effect is NOT MEASURED;
* every downstream renderer is None-aware: the report writer, handed an empty arm, does not raise
  and writes the absence word rather than a number.

All inputs are synthetic. ``scripts/run_grid_ablation`` imports ``backtests.backtest_grid_replay``
→ ``super_bot`` (side-effecting module bodies), so the fixture chdirs into ``tmp_path`` BEFORE the
import. The writer is only ever called under the monkeypatched cwd, so nothing can land in the
repo's ``audits/``; no test here reads a production ``*.db``, ``data/historical``, or the network.
"""

from pathlib import Path

import pytest


@pytest.fixture
def mod(monkeypatch, tmp_path):
    """The script under test, imported with the cwd inside tmp_path (super_bot safety)."""
    monkeypatch.chdir(tmp_path)
    import scripts.run_grid_ablation as rg
    return rg


def _trade(pnl, side="LONG", adx=22.0, symbol="BTCUSDT"):
    return {
        "side": side,
        "net_pnl": float(pnl),
        "is_maker_exit": True,
        "holding_mins": 10.0,
        "entry_fee_usd": 0.01,
        "exit_fee_usd": 0.01,
        "funding_usd": 0.0,
        "entry_market_adx": float(adx),
        "symbol": symbol,
    }


# ---------------------------------------------------------------------------
# Pin 1: the empty arm publishes absences, not measured-looking zeros.
# ---------------------------------------------------------------------------

def test_empty_arm_publishes_absences_not_zeros(mod):
    m = mod.compute_metrics([])
    assert m["n_trades"] == 0, "the trade count IS the measurement and must stay"
    assert m["win_rate"] is None
    assert m["profit_factor"] is None
    assert m["ev_mean"] is None
    assert m["ev_ci_90"] is None, "the CI must be the absence, not a fabricated (0.0, 0.0)"
    reason = m["absence_reason"]
    assert isinstance(reason, str) and reason, "the absence must carry a reason string"
    assert "no trades" in reason.lower()


def test_bootstrap_ci_empty_sample_publishes_absence(mod):
    assert mod.bootstrap_ci([]) == (None, None, None)


# ---------------------------------------------------------------------------
# Pin 2 & 3: profit_factor — no denominator is an absence; a mixed sample is unchanged.
# ---------------------------------------------------------------------------

def test_no_losing_trade_profit_factor_is_none(mod):
    m = mod.compute_metrics([_trade(5.0), _trade(10.0), _trade(0.5)])
    assert m["profit_factor"] is None, "no denominator must never publish a stand-in like 99.0"
    assert m["win_rate"] == 100.0, "the measured win rate must stay a reading"
    assert m["ev_mean"] is not None and m["ev_mean"] > 0.0


def test_mixed_sample_profit_factor_is_the_measured_ratio(mod):
    # gross profit 15.0 over gross loss 13.0 -> 1.1538461538461537 (written by hand, unchanged
    # from the pre-fix measured path: the fix only replaces the no-denominator branches).
    m = mod.compute_metrics([_trade(10.0), _trade(5.0), _trade(-5.0), _trade(-5.0), _trade(-3.0)])
    assert m["profit_factor"] == pytest.approx(1.1538461538461537)
    assert m["win_rate"] == pytest.approx(40.0)
    assert m["ev_mean"] == pytest.approx((10 + 5 - 5 - 5 - 3) / 5)


# ---------------------------------------------------------------------------
# The three Section-1 verdicts are derived from the row's own numbers.
# ---------------------------------------------------------------------------

def test_win_rate_verdict_is_derived_and_carries_no_number(mod):
    assert "NOT MEASURED" in mod._win_rate_verdict(None)
    supported = mod._win_rate_verdict(90.0)
    refuted = mod._win_rate_verdict(46.2)
    assert "SUPPORTED" in supported and "REFUTED" not in supported
    assert "REFUTED" in refuted and "SUPPORTED" not in refuted
    for verdict in (supported, refuted):
        assert _has_no_hardcoded_figure(verdict), (
            f"no hard-coded number may appear in a verdict cell: {verdict!r}")
    assert "46.2" not in refuted, "the old fabricated '~46.2%' must not survive in any phrasing"


def test_ev_verdict_is_derived_from_the_intervals_own_sign(mod):
    straddle = mod._ev_verdict({"ev_mean": 0.1, "ev_ci_90": (-0.2, 0.3)})
    assert "straddles $0" in straddle
    positive = mod._ev_verdict({"ev_mean": 0.5, "ev_ci_90": (0.2, 0.8)})
    assert "excludes zero" in positive and "straddles" not in positive
    negative = mod._ev_verdict({"ev_mean": -0.5, "ev_ci_90": (-0.8, -0.2)})
    assert "NEGATIVE" in negative and "excludes zero" in negative
    absent = mod._ev_verdict({"ev_mean": None, "ev_ci_90": None})
    assert "NOT MEASURED" in absent
    for verdict in (straddle, positive, negative, absent):
        assert _has_no_hardcoded_figure(verdict), (
            f"no hard-coded number may appear in a verdict cell: {verdict!r}")


# ---------------------------------------------------------------------------
# Pin 4: the report writer, handed an empty arm, does not raise and writes the absence word.
# ---------------------------------------------------------------------------

def _results(mod, *, deep_longs):
    """A results dict shaped exactly like run_full_suite() builds, with the deep-chop arm
    switchable to empty so the writer's absence paths are exercised."""
    mid_longs = [_trade(1.0, adx=22.0), _trade(-0.5, adx=24.0), _trade(2.0, adx=21.0)]
    upper_longs = [_trade(-0.2, adx=26.0)]
    shorts = [_trade(0.9, side="SHORT"), _trade(-0.4, side="SHORT"), _trade(0.3, side="SHORT")]
    all_trades = deep_longs + mid_longs + upper_longs + shorts

    per_symbol = {}
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        sym_trades = [dict(t, symbol=sym) for t in all_trades]
        split = mod.split_engine_metrics(sym_trades, signal_to_fill_pct=61.0)
        longs = [t for t in sym_trades if t["side"] == "LONG"]
        per_symbol[sym] = {
            "conservative": split,
            "optimistic": split,
            "ttl_6bars": split,
            "no_regime": split,
            "no_vwap": split,
            "no_45m": split,
            "funding_gate": split,
            "stratified_long": {
                "deep_chop_lt20": mod.compute_metrics([t for t in longs if t["entry_market_adx"] < 20.0]),
                "mid_chop_20_25": mod.compute_metrics([t for t in longs if 20.0 <= t["entry_market_adx"] < 25.0]),
                "upper_chop_25_28": mod.compute_metrics([t for t in longs if 25.0 <= t["entry_market_adx"] < 28.0]),
            },
            "fleet_capped_long": mod.compute_metrics(longs),
            "fleet_deep_chop": mod.compute_metrics([t for t in longs if t["entry_market_adx"] < 20.0]),
        }

    return {
        "fleet_cap2": {"trades": all_trades, "signal_to_fill_pct": 61.0},
        "fleet_ceil20": {"trades": list(deep_longs), "signal_to_fill_pct": 61.0},
        "fleet_cap99": {"trades": all_trades + [_trade(-0.7, adx=18.0)], "signal_to_fill_pct": 70.0},
        "per_symbol": per_symbol,
    }


def _verdict_cell(row: str) -> str:
    return [c.strip() for c in row.strip().strip("|").split("|")][-1]


def _has_no_hardcoded_figure(cell: str) -> bool:
    # "$0" is the threshold the claim itself names (the claim column, not a measured figure);
    # every other digit in a verdict cell would be a number this row did not measure.
    return not any(ch.isdigit() for ch in cell.replace("$0", ""))


def _write(mod, results, tag):
    path = mod.generate_report(results, tag)
    p = Path(path)
    assert p.parent.name == "audits"
    return p


def test_writer_handed_empty_arm_writes_absence_word_and_does_not_raise(mod, tmp_path, capsys):
    results = _results(mod, deep_longs=[])
    path = _write(mod, results, "EMPTYARM")  # cwd is tmp_path via the fixture -> tmp_path/audits/
    assert str(tmp_path) in str(path.resolve()), "a test must never be able to write into repo audits/"
    text = path.read_text(encoding="utf-8")

    row_a = next(ln for ln in text.splitlines() if ln.startswith("| (a)"))
    row_b = next(ln for ln in text.splitlines() if ln.startswith("| (b)"))
    assert "not measured" in row_a, "an empty arm must render the absence word, not 0.0"
    assert "**0.0% WR**" not in row_a and "| 0.0% |" not in row_a
    assert "NOT MEASURED" in row_a and "this arm closed no trades" in row_a
    assert "NOT MEASURED" in row_b
    assert "no trades in this arm" in row_b
    # the Section-2/4/6/7 renderers all met None and survived:
    assert "TypeError" not in capsys.readouterr().out


def test_writer_derives_verdicts_and_no_hardcoded_number_survives(mod, tmp_path):
    # A below-claim win rate (2 of 4 = 50%) and a CI that straddles zero.
    deep = [_trade(0.7, adx=15.0), _trade(-0.4, adx=18.0), _trade(0.6, adx=19.0), _trade(-0.5, adx=17.0)]
    results = _results(mod, deep_longs=deep)
    path = _write(mod, results, "MEASURED")
    assert str(tmp_path) in str(path.resolve())
    text = path.read_text(encoding="utf-8")

    row_a = next(ln for ln in text.splitlines() if ln.startswith("| (a)"))
    row_b = next(ln for ln in text.splitlines() if ln.startswith("| (b)"))
    row_c = next(ln for ln in text.splitlines() if ln.startswith("| (c)"))

    assert "50.0% WR" in row_a and "REFUTED" in row_a
    assert _has_no_hardcoded_figure(_verdict_cell(row_a)), _verdict_cell(row_a)

    assert "MARGINAL / NOT DEMONSTRATED" in row_b and "straddles $0" in row_b
    assert _has_no_hardcoded_figure(_verdict_cell(row_b)), _verdict_cell(row_b)

    assert "NOT MEASURED IN THIS RUN" in row_c
    assert "does not measure the ADX shield" in row_c

    # the three fabricated cells are gone from the report itself...
    assert "CONFIRMED SHIELD" not in text
    assert "Peak WR" not in text
    assert "-0.064" not in text
    assert "46.2" not in text


def test_script_source_no_longer_contains_the_fabricated_literals():
    src = (Path(__file__).resolve().parent.parent / "scripts" / "run_grid_ablation.py").read_text(
        encoding="utf-8")
    for literal in ("Peak WR ~46.2%", "CONFIRMED SHIELD", "Net EV ~ -0.064%",
                    "99.0 if gross_profit"):
        assert literal not in src
