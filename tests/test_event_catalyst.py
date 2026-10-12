import asyncio
import json
import logging
import ai_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

async def test_catalyst():
    # 1. Test Geopolitical War Escalation (Expected: STRUCTURAL_MACRO_BEAR / SECULAR_SHORT_BREAKDOWN)
    war_event = {
        "headline": "BREAKING: Regional military alliance launches nationwide strikes; global oil prices surge 7%, emergency UN security council called",
        "timestamp": "2026-09-28T22:30:00Z",
        "btc_15m_velocity_pct": -3.2,
        "spy_1h_change_pct": -1.8,
        "dxy_index": 105.8,
        "bitget_funding_rate_8h": 0.005,
        "bitget_open_interest_delta_usd": 150000000.0,
        "recent_price_reaction": "Heavy sustained selling through orderbook bids with rising volume"
    }

    print("\n--- Testing Geopolitical War Event ---")
    res_war = await ai_engine.evaluate_event_catalyst(war_event, timeout_sec=12.0)
    print(json.dumps(res_war, indent=2))
    assert res_war is not None
    assert "event_category" in res_war
    assert "expected_horizon" in res_war
    assert "tactical_playbook" in res_war
    assert "directive" in res_war

    # 2. Test Flash Liquidity Shock / Fakeout Rumor (Expected: FLASH_LIQUIDITY_SHOCK / FLASH_REBOUND_TRAP)
    flash_event = {
        "headline": "Unverified Telegram rumor claims major exchange wallet halted withdrawals; CT influencers panic",
        "timestamp": "2026-09-28T22:35:00Z",
        "btc_15m_velocity_pct": -2.9,
        "spy_1h_change_pct": 0.1,
        "dxy_index": 103.2,
        "bitget_funding_rate_8h": -0.045,
        "bitget_open_interest_delta_usd": -380000000.0,
        "recent_price_reaction": "Sudden long liquidation cascade wicking into liquidity, funding deeply negative"
    }

    print("\n--- Testing Flash Liquidity Shock Event ---")
    res_flash = await ai_engine.evaluate_event_catalyst(flash_event, timeout_sec=12.0)
    print(json.dumps(res_flash, indent=2))
    assert res_flash is not None
    assert "tactical_playbook" in res_flash

    print("\n[SUCCESS] AI Event Catalyst & Horizon Arbiter verified successfully!")

if __name__ == "__main__":
    asyncio.run(test_catalyst())
