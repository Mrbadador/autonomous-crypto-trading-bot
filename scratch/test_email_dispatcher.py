import sys
import os

# Add codebase to path
sys.path.append(r"c:\Users\user\.gemini\antigravity\scratch\llama_trading_bot")

from email_dispatcher import send_trade_log

def test_email_dispatcher():
    print("[TEST] Running dynamic column resolution query test for email_dispatcher...")
    
    # We will invoke send_trade_log with dummy data. We don't want it to actually dispatch SMTP mail
    # since we don't have active mail configuration.
    # Let's mock smtplib.SMTP inside email_dispatcher to avoid real network attempts.
    import smtplib
    class DummySMTP:
        def __init__(self, *args, **kwargs): pass
        def starttls(self): pass
        def login(self, *args, **kwargs): pass
        def sendmail(self, *args, **kwargs): pass
        def quit(self): pass
        
    smtplib.SMTP = DummySMTP
    
    # Run dynamic lookup on existing database columns (which we migrated in earlier steps)
    # This proves no sqlite3.OperationalError occurs when querying columns
    try:
        send_trade_log(
            symbol="BTC/USDT",
            side="long",
            price=50000.0,
            pnl=0.0,
            trade_context={"qty": 1.0, "regime": "BULLISH"}
        )
        print("[TEST] Dynamic query executed cleanly without database crashes!")
    except Exception as e:
        print(f"[TEST] Dynamic query test encountered error: {e}")
        assert False, f"Dynamic query resolution failed: {e}"

    print("\n[SUCCESS] Email Dispatcher Dynamic Database Resolution PASSED!")

if __name__ == "__main__":
    test_email_dispatcher()
