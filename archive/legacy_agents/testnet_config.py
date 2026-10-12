import os
import config

# Enforce Paper/Live Mode
config.MODE = "paper"
config.IS_REPLAY = False
config.SANDBOX = True # Critical for Testnet

# Strictly enforce Testnet Wallet Credentials
config.BINANCE_API_KEY = os.getenv("TESTNET_API_KEY", "")
config.BINANCE_SECRET_KEY = os.getenv("TESTNET_SECRET_KEY", "")

# Extra Balance Protection: Never read Mainnet Keys
mainnet_key = os.getenv("MAINNET_API_KEY", "")
if config.BINANCE_API_KEY == mainnet_key and mainnet_key != "":
    raise ValueError("CRITICAL: testnet_config.py is reading MAINNET keys! Aborting.")
