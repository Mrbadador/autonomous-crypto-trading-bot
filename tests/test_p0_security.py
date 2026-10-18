import os
import stat
import tempfile
import unittest
from unittest.mock import patch


class TestCredentialValidation(unittest.TestCase):
    def test_world_readable_env_rejected(self):
        """Verify that world-readable .env files cause startup failure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("BITGET_API_KEY=test\n")
            temp_env = f.name
        
        try:
            # Set world-readable permissions
            os.chmod(temp_env, 0o644)
            
            # Import and test the validation function
            from credential_validator import validate_credentials
            
            # Patch the env path to use our temp file
            with patch('credential_validator.os.path.join', return_value=temp_env), \
                 patch('credential_validator.os.path.exists', return_value=True), \
                 patch('credential_validator.os.stat') as mock_stat:
                mock_stat.return_value = os.stat_result((0o100644, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                
                result = validate_credentials()
                
                self.assertFalse(result, "World-readable .env should fail validation")
        finally:
            os.remove(temp_env)
    
    def test_missing_credentials_rejected(self):
        """Verify that missing required credentials cause startup failure."""
        from credential_validator import validate_credentials
        
        with patch.dict(os.environ, {}, clear=True):
            result = validate_credentials()
            self.assertFalse(result, "Missing credentials should fail validation")
    
    def test_valid_credentials_pass(self):
        """Verify that valid credentials pass validation."""
        from credential_validator import validate_credentials
        
        with patch.dict(os.environ, {
            "BITGET_API_KEY": "test_key",
            "BITGET_API_SECRET": "test_secret",
            "BITGET_PASSPHRASE": "test_pass",
            "GMAIL_USER": "test@example.com",
            "GMAIL_APP_PASSWORD": "test_password"
        }):
            with patch('credential_validator.os.path.exists', return_value=False):
                result = validate_credentials()
                self.assertTrue(result, "Valid credentials with secure permissions should pass")


if __name__ == '__main__':
    unittest.main()
