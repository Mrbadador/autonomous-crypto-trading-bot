import testnet_config
import asyncio
# NOTE: Legacy testnet runner — NOT part of the V11.0 live stack.
# Use main.py for live trading. Archived bot logic: archive/bot_v10_backtest.py
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "archive"))
from bot_v10_backtest import main

if __name__ == "__main__":
    print("=" * 60)
    print("   LAUNCHING LLAMA TRADING ENGINE [TESTNET MODE] ")
    print("=" * 60)
    print("Mode: PAPER / SANDBOX")
    print("Engine: Starting...")
    print("=" * 60 + "\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown via Ctrl+C. Exiting...")
