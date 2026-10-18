"""Credential validation utilities for startup security checks."""
import os
import stat
import logging

log = logging.getLogger("credential_validator")


def validate_credentials() -> bool:
    """Validate that critical credentials are loaded and warn if .env is world-readable.
    
    Returns True if all checks pass, False otherwise.
    """
    # Check .env file permissions
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        try:
            file_stat = os.stat(env_path)
            mode = file_stat.st_mode
            # Check if others have read permission (world-readable)
            if mode & stat.S_IROTH:
                log.critical("[SECURITY] .env file is world-readable! Run: chmod 600 .env")
                return False
        except Exception as perm_err:
            log.warning(f"[SECURITY] Could not check .env permissions: {perm_err}")
    
    # Validate required credentials are present
    required_keys = [
        "BITGET_API_KEY", "BITGET_API_SECRET", "BITGET_PASSPHRASE",
        "GMAIL_USER", "GMAIL_APP_PASSWORD"
    ]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        log.critical(f"[SECURITY] Missing required credentials: {missing}")
        return False
    
    log.info("[SECURITY] Credential validation passed")
    return True
