"""
ENV Integrity Dry-Run Test
Tests that load_dotenv() and BINANCE_API_KEY are correctly wired in all modules.
"""
import os
import sys

# Must run from the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '.')

print('=' * 65)
print('  ENV CUSTODIAN — DRY-RUN INTEGRITY SCAN')
print('=' * 65)

# -----------------------------------------------------------------------
# TEST 1: load_dotenv() from local .env
# -----------------------------------------------------------------------
from dotenv import load_dotenv
dotenv_local = os.path.join(os.path.abspath('.'), '.env')
result = load_dotenv(dotenv_path=dotenv_local, verbose=True)
print(f'\n[TEST 1] load_dotenv() path: {dotenv_local}')
print(f'         .env file found & loaded: {result}')

# -----------------------------------------------------------------------
# TEST 2 & 3: Key presence in os.environ
# -----------------------------------------------------------------------
api_key    = os.getenv('BINANCE_API_KEY', '')
secret_key = os.getenv('BINANCE_SECRET_KEY', '')

print(f'\n[TEST 2] BINANCE_API_KEY   : ', end='')
if api_key:
    print(f'PRESENT (len={len(api_key)}, last4=...{api_key[-4:]})')
else:
    print('MISSING / EMPTY')

print(f'[TEST 3] BINANCE_SECRET_KEY: ', end='')
if secret_key:
    print(f'PRESENT (len={len(secret_key)}, last4=...{secret_key[-4:]})')
else:
    print('MISSING / EMPTY')

# -----------------------------------------------------------------------
# TEST 4: config.py round-trip
# -----------------------------------------------------------------------
print('\n[TEST 4] Importing config.py ...')
try:
    import config
    cfg_key = config.BINANCE_API_KEY
    print(f'         config.py import OK.')
    if cfg_key:
        print(f'         config.BINANCE_API_KEY PRESENT (len={len(cfg_key)}, last4=...{cfg_key[-4:]})')
    else:
        print('         config.BINANCE_API_KEY MISSING/EMPTY in config — check that load_dotenv() is called before config is imported.')
except Exception as e:
    print(f'         config.py import FAILED: {e}')

# -----------------------------------------------------------------------
# TEST 5: Static source scan — verify each module has the required wiring
# -----------------------------------------------------------------------
print('\n[TEST 5] Static source scan for load_dotenv() wiring:')
print('-' * 65)

modules_check = [
    # (filename, needs_env_propagate)
    ('config.py',           False),
    ('recovery_agent.py',   False),
    ('marl_live_trader.py', True),   # also spawns a subprocess
    ('live_oracle.py',      False),
    ('regime_classifier.py',False),
    ('main.py',             True),   # spawns ALL subprocesses
    ('start_system.py',     True),   # secondary ecosystem launcher
]

all_pass = True
for fname, needs_propagate in modules_check:
    try:
        with open(fname, 'r', encoding='utf-8') as f:
            src = f.read()
    except FileNotFoundError:
        print(f'  [SKIP] {fname} not found')
        continue

    has_import   = 'from dotenv import load_dotenv' in src
    has_call     = 'load_dotenv(' in src
    has_key_ref  = 'BINANCE_API_KEY' in src
    has_env_prop = ('env=os.environ' in src) if needs_propagate else True

    ok = has_import and has_call and has_key_ref and has_env_prop
    all_pass = all_pass and ok
    badge = 'PASS' if ok else 'FAIL'
    prop_note = f', env_propagate={has_env_prop}' if needs_propagate else ''
    print(f'  [{badge}] {fname:<25} dotenv_import={has_import}, load_dotenv={has_call}, key_ref={has_key_ref}{prop_note}')

# -----------------------------------------------------------------------
# FINAL VERDICT
# -----------------------------------------------------------------------
print('\n' + '=' * 65)
if api_key and all_pass:
    print('  [OK] RESULT: ALL CHECKS PASSED.')
    print(f'       BINANCE_API_KEY is loaded and all modules are correctly wired.')
elif not api_key:
    print('  [FAIL] RESULT: BINANCE_API_KEY NOT FOUND in environment.')
    print('         Action: Verify your .env file contains BINANCE_API_KEY=...')
    print(f'         Expected .env path: {dotenv_local}')
else:
    print('  [WARN] RESULT: KEY PRESENT but some source wiring checks FAILED.')
    print('         Review the FAIL lines above.')
print('=' * 65)
