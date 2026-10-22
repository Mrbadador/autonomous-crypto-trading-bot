import pytest


@pytest.mark.parametrize(
    "reason, pnl, should_cooldown",
    [
        ("STOP_LOSS", -1.25, True),
        ("AI_STRUCTURAL_INVALIDATION", -3.0, True),
        ("SCALP_SHORT_TIME_EXPIRY", -0.5, True),
        ("STOP_LOSS", 0.75, False),
        ("sl_hit", 0.10, False),
        ("breakeven_hit", 0.0, False),
        ("TAKE_PROFIT", 2.5, False),
    ],
)
def test_cooldown_predicate_is_pnl_keyed(reason, pnl, should_cooldown):
    from swing_trader import _cooldown_should_trigger

    assert _cooldown_should_trigger(reason, pnl) is should_cooldown
