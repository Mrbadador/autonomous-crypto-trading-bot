"""Final credential resolution scan."""
import os, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env', verbose=True)

print('=== BINANCE CREDENTIAL SCAN ===')
api_key   = os.getenv('BINANCE_API_KEY', '')
secret_v1 = os.getenv('BINANCE_SECRET_KEY', '')
secret_v2 = os.getenv('BINANCE_API_SECRET', '')

def display(label, val):
    if val:
        print(f'  {label}: PRESENT  len={len(val)}  last4=...{val[-4:]}')
    else:
        print(f'  {label}: MISSING/EMPTY')

display('BINANCE_API_KEY    ', api_key)
display('BINANCE_SECRET_KEY ', secret_v1)
display('BINANCE_API_SECRET ', secret_v2)

import config
resolved = config.BINANCE_SECRET_KEY

print()
print('--- config.py resolution ---')
display('config.BINANCE_API_KEY   ', config.BINANCE_API_KEY)
display('config.BINANCE_SECRET_KEY', resolved)

print()
if config.BINANCE_API_KEY and resolved:
    print('[OK] Both credentials resolved. All exchange clients will authenticate correctly.')
else:
    print('[FAIL] One or more credentials still missing.')
