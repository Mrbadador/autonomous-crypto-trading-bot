"""
tests/test_inflight_shepherd_lock_reason.py — F5 measurement fidelity.

The lock DECISION in inflight_shepherd was correct; the LABEL named a cause the
measurement does not support ("is actively managed by swing_architect" printed for
every fail-closed path). These tests pin the lifted module-level
inflight_shepherd._is_swing_locked (locked, reason) contract, reachable WITHOUT
any database through the injectable lock_reader, and pin the caller-unpack so the
always-truthy tuple can never re-enter the advisory branch on (False, reason).
"""

import inspect
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def shepherd(monkeypatch, tmp_path):
    """chdir(tmp_path) BEFORE importing inflight_shepherd: its module body runs
    load_dotenv, sys.stdout.reconfigure, and _ensure_shepherd_tables() against a
    relative trading_ledger.db — all must land in tmp_path, never live paths."""
    monkeypatch.chdir(tmp_path)
    import inflight_shepherd
    return inflight_shepherd


def _ts(offset_seconds):
    return (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()


# ---------------------------------------------------------------------------
# The five real causes
# ---------------------------------------------------------------------------

def test_cause_a_no_row_locked_with_no_row_reason(shepherd):
    locked, reason = shepherd._is_swing_locked("BTC/USDT", lock_reader=lambda sym: None)
    assert locked is True
    assert "no asset_locks row" in reason
    assert "BTC/USDT" in reason


@pytest.mark.parametrize("fleet_holder", ["swing_architect", "super_bot_grid", "super_bot_scalp_short"])
def test_cause_b_fleet_holder_locked_with_holder_name(shepherd, fleet_holder):
    locked, reason = shepherd._is_swing_locked(
        "ETH/USDT", lock_reader=lambda sym, h=fleet_holder: (h, _ts(30))
    )
    assert locked is True
    assert reason == f"held by {fleet_holder}"


def test_cause_c_fresh_heartbeat_from_non_fleet_holder_is_locked(shepherd):
    locked, reason = shepherd._is_swing_locked(
        "BTC/USDT", lock_reader=lambda sym: ("external_shepherd", _ts(10))
    )
    assert locked is True
    assert "fresh heartbeat" in reason
    assert "external_shepherd" in reason
    assert "s old" in reason


def test_cause_d_unreadable_lock_state_fails_closed(shepherd):
    def exploding_reader(sym):
        raise RuntimeError("database is locked")

    locked, reason = shepherd._is_swing_locked("BTC/USDT", lock_reader=exploding_reader)
    assert locked is True
    assert "unreadable" in reason
    assert "failing closed" in reason
    assert "database is locked" in reason


def test_cause_f_stale_heartbeat_is_the_only_false_path(shepherd):
    locked, reason = shepherd._is_swing_locked(
        "BTC/USDT", lock_reader=lambda sym: ("external_shepherd", _ts(3600))
    )
    assert locked is False
    assert "stale heartbeat" in reason
    assert "external_shepherd" in reason


# ---------------------------------------------------------------------------
# The whole point: the four True reasons name different causes
# ---------------------------------------------------------------------------

def test_the_four_locked_reasons_are_pairwise_distinct(shepherd):
    def boom(sym):
        raise RuntimeError("contention")

    cases = [
        shepherd._is_swing_locked("BTC/USDT", lock_reader=lambda s: None)[1],
        shepherd._is_swing_locked("BTC/USDT", lock_reader=lambda s: ("swing_architect", _ts(30)))[1],
        shepherd._is_swing_locked("BTC/USDT", lock_reader=lambda s: ("external_shepherd", _ts(30)))[1],
        shepherd._is_swing_locked("BTC/USDT", lock_reader=boom)[1],
    ]
    assert len(cases) == len(set(cases)) == 4, f"locked reasons must name distinct causes: {cases}"


# ---------------------------------------------------------------------------
# Decision-mapping pins: True/False behaviour is unchanged from the old closure
# ---------------------------------------------------------------------------

def test_900s_boundary_decision_unchanged(shepherd):
    # Old code: return (now - hb).total_seconds() <= 900.0 — the bound itself is
    # inclusive; pin the comparison in source and the mapping either side of it
    # (wall-clock drift makes an exact-900.0 runtime probe race-prone).
    src = inspect.getsource(shepherd._is_swing_locked)
    assert "age <= 900.0" in src
    fresh, _ = shepherd._is_swing_locked("X/USDT", lock_reader=lambda s: ("ext", _ts(899)))
    stale, _ = shepherd._is_swing_locked("X/USDT", lock_reader=lambda s: ("ext", _ts(901)))
    assert fresh is True
    assert stale is False


def test_bad_timestamp_fails_closed_like_the_original(shepherd):
    # Original: fromisoformat("") raised inside the try and returned True (fail-closed).
    locked, reason = shepherd._is_swing_locked("X/USDT", lock_reader=lambda s: ("ext", ""))
    assert locked is True
    assert "unreadable" in reason


# ---------------------------------------------------------------------------
# Caller-consumption pin: the fail-open tuple bug can never come back
# ---------------------------------------------------------------------------

def test_raw_tuple_is_truthy_so_caller_must_unpack(shepherd):
    # Pin the hazard: (False, reason) as a raw tuple is truthy. If the caller ever
    # tests the tuple itself, a ratchet-approved (False) result would take the
    # advisory branch and the fail-closed check would fail-OPEN.
    approved = (False, "stale heartbeat from external_shepherd")
    assert bool(approved) is True, "a tuple is always truthy — never test the raw return"
    swing_is_locked, _ = approved
    assert swing_is_locked is False, "the unpacked bool is the decision"

    src = inspect.getsource(shepherd)
    assert "swing_is_locked, lock_reason = await asyncio.to_thread(_is_swing_locked, std_sym)" in src
    assert "if swing_is_locked:" in src
    # The old raw-value names must be gone from the caller.
    assert "swing_is_owner" not in src
    assert "if _is_swing_locked" not in src


def test_advisory_label_names_the_cause_not_an_assumption(shepherd):
    src = inspect.getsource(shepherd)
    # The unsupported claim is gone.
    assert "is actively managed by swing_architect" not in src
    assert 'f"Advisory Shepherding: {verdict}"' not in src
    # Both the log line and the persisted diagnosis carry lock_reason.
    assert "not ratcheting ({lock_reason})" in src
    assert "Advisory Shepherding ({lock_reason})" in src


def test_absent_verdict_never_prints_none(shepherd):
    src = inspect.getsource(shepherd)
    # House convention: an absence renders as an absence word, never the literal "None".
    assert 'verdict if verdict else "not measured"' in src


def test_default_reader_runs_the_same_production_query(shepherd):
    src = inspect.getsource(shepherd._default_lock_reader)
    assert 'sqlite3.connect("trading_data.db", timeout=5.0)' in src
    assert "SELECT locked_by, heartbeat FROM asset_locks WHERE symbol = ?" in src
    # And it is the reader wired in when none is injected.
    assert "lock_reader or _default_lock_reader" in inspect.getsource(shepherd._is_swing_locked)
