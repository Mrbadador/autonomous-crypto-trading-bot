import os
import unittest


class TestMobileBridgeSecurity(unittest.TestCase):
    def test_missing_password_raises_error(self):
        """Verify mobile bridge refuses to start without APP_PASSWORD."""
        with self.assertRaises(EnvironmentError) as ctx:
            # Force reload to trigger the check at module level
            import archive.legacy_agents.agent_mobile_bridge as bridge
            import importlib
            importlib.reload(bridge)
        
        self.assertIn("APP_PASSWORD", str(ctx.exception))
    
    def test_hardcoded_password_removed(self):
        """Verify no hardcoded fallback password exists in source."""
        with open("archive/legacy_agents/agent_mobile_bridge.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        self.assertNotIn("SovereignSwarm2026!", content,
                        "Hardcoded password still present in source")
    
    def test_cloudflare_tunnel_disabled(self):
        """Verify Cloudflare Tunnel launch is commented out in start script."""
        with open("scripts/start_mobile_bridge.py", "r") as f:
            content = f.read()
        
        # Check that tunnel command is disabled
        self.assertIn("[SECURITY] Cloudflare Tunnel disabled", content,
                     "Cloudflare Tunnel should be disabled with security warning")
        
        # Verify localhost binding
        self.assertIn('"--host", "127.0.0.1"', content,
                     "Should bind to localhost only, not 0.0.0.0")
    
    def test_password_not_printed_to_stdout(self):
        """Verify APP_PASSWORD is never printed to terminal."""
        with open("scripts/start_mobile_bridge.py", "r") as f:
            content = f.read()
        
        # Should not contain direct password printing
        self.assertNotIn('print(f"  App Password  : {APP_PASSWORD}', content,
                        "Password should not be printed to stdout")
        
        # Should have safe message instead
        self.assertIn("(set in .env - never share this)", content,
                     "Should display safe placeholder message")


if __name__ == '__main__':
    unittest.main()
