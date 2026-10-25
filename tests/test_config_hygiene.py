"""
tests/test_config_hygiene.py

AST-based static validation suite to guarantee configuration hygiene in config.py:
1. Zero module-level duplicate variable assignments.
2. Verified single source of truth for all risk, sizing, and execution parameters.
"""

import ast
import os
import unittest


class TestConfigHygiene(unittest.TestCase):
    def test_zero_module_level_duplicate_assignments_in_config(self):
        """
        Parses config.py and verifies that no top-level configuration variable
        is assigned more than once, preventing silent 'last assignment wins' bugs.
        """
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
        self.assertTrue(os.path.exists(config_path), f"config.py not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename="config.py")

        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments.setdefault(target.id, []).append(node.lineno)

        duplicates = {var: lines for var, lines in assignments.items() if len(lines) > 1}
        self.assertEqual(
            duplicates,
            {},
            f"Found duplicate variable assignments in config.py: {duplicates}. "
            f"Every configuration setting must have exactly one assignment."
        )

    def test_grid_exit_economics_calibrated_values(self):
        """
        Asserts that the critical exit economics parameters in config.py
        have their intended values to protect against premature winner choking.
        """
        import config
        # GRID_BREAKEVEN_TRIGGER_PCT must be at least 1.20% (relaxed from 0.75% to prevent micro-clipping)
        self.assertGreaterEqual(config.GRID_BREAKEVEN_TRIGGER_PCT, 0.0120, "GRID_BREAKEVEN_TRIGGER_PCT must be >= 1.20%")
        # Target TP must exceed SL
        self.assertGreater(config.GRID_TP_PCT, config.GRID_SL_PCT, "GRID_TP_PCT must exceed GRID_SL_PCT")

    # ------------------------------------------------------------------
    # F4 hygiene: every module-level GRID_* config key must have a REAL
    # consumer, or be explicitly declared (and documented) as unused.
    # This catches the class of defect where a config comment/docstring
    # claims an effect that the code does not actually read.
    # ------------------------------------------------------------------

    # Documented known-unused GRID_* keys: each of these has NO engine
    # consumer, so setting it has no effect (see config.py's own comments).
    # This is the ONLY allowed exception to the "must be consumed" rule, and
    # test_known_unused_grid_set_membership_pinned below pins its exact
    # membership so the set cannot silently drift.
    _KNOWN_UNUSED_GRID_KEYS = {
        "GRID_RSI_SHORT_THRESHOLD",  # dead: no consumer; live overbought read is SCALP_SHORT_RSI_OVERBOUGHT
        "GRID_BB_STD",               # dead: no consumer; live band is 2.0 sigma over 24 bars
        "GRID_VWAP_WINDOW",          # dead: no consumer; live VWAP window is 24 bars
    }

    # Directories that are NOT active modules and must be excluded from the
    # consumer scan: dead archive, virtualenvs, a vendored interpreter tree,
    # node_modules, VCS caches, and other git worktrees checked out under .qoder.
    _GRID_SCAN_IGNORE_DIRS = {
        "archive", ".venv", "venv", "Python", "node_modules",
        ".git", "__pycache__", ".qoder",
    }

    def _repo_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _collect_grid_names(self):
        """Parse config.py as TEXT (AST, no import) and collect module-level GRID_* assignment targets."""
        config_path = os.path.join(self._repo_root(), "config.py")
        with open(config_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename="config.py")
        names = []
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("GRID_"):
                        names.append(target.id)
        return names

    def _scan_grid_consumers(self, grid_names):
        """
        Search active modules (never config.py or this test file) for an ACTUAL
        consumer pattern per key -- ``config.<NAME>`` or
        ``getattr(config, "<NAME>"`` / ``getattr(config, '<NAME>'`` -- NOT a bare
        substring. A bare-substring scan would be wrong: after fix (1) config.py's
        new comments and this file's _KNOWN_UNUSED_GRID_KEYS literal both contain
        the key names as plain text, so those files are excluded and only the
        qualified consumer patterns are matched. Returns the set of consumed names.
        """
        root = self._repo_root()
        needles = {}
        for name in grid_names:
            needles[name] = (
                "config." + name,
                'getattr(config, "%s"' % name,
                "getattr(config, '%s'" % name,
            )
        skip_files = {
            os.path.normcase(os.path.join(root, "config.py")),
            os.path.normcase(os.path.abspath(__file__)),
        }
        consumed = set()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self._GRID_SCAN_IGNORE_DIRS]
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                if os.path.normcase(full) in skip_files:
                    continue
                try:
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                except OSError:
                    continue
                for name, pats in needles.items():
                    if name in consumed:
                        continue
                    if any(p in text for p in pats):
                        consumed.add(name)
        return consumed

    def test_all_grid_keys_consumed_or_declared_unused(self):
        """
        Every module-level GRID_* key must be either consumed by an active module
        or present in _KNOWN_UNUSED_GRID_KEYS. Falsifiable: a new unconsumed
        GRID_FOO that is not declared unused fails here.
        """
        grid_names = self._collect_grid_names()
        self.assertTrue(grid_names, "Expected module-level GRID_* keys in config.py")
        consumed = self._scan_grid_consumers(grid_names)
        undeclared = [
            n for n in grid_names
            if n not in consumed and n not in self._KNOWN_UNUSED_GRID_KEYS
        ]
        self.assertEqual(
            undeclared,
            [],
            "GRID_* keys with no real consumer and not declared in "
            "_KNOWN_UNUSED_GRID_KEYS: %s. Either wire the key into an engine, "
            "delete it, or add it to the documented known-unused set." % undeclared,
        )

    def test_known_unused_grid_set_membership_pinned(self):
        """
        Pin the known-unused set so it cannot drift:
        1. the three documented-dead keys must be present and genuinely unconsumed;
        2. no key that actually has a consumer may sneak into the set;
        3. the set must reference only real GRID_* names from config.py.
        """
        grid_names = set(self._collect_grid_names())
        consumed = self._scan_grid_consumers(sorted(grid_names))

        for dead in ("GRID_RSI_SHORT_THRESHOLD", "GRID_BB_STD", "GRID_VWAP_WINDOW"):
            self.assertIn(dead, self._KNOWN_UNUSED_GRID_KEYS,
                          "%s must stay declared in _KNOWN_UNUSED_GRID_KEYS" % dead)
            self.assertNotIn(dead, consumed,
                             "%s unexpectedly has a consumer; it is not dead" % dead)

        leaked = sorted(self._KNOWN_UNUSED_GRID_KEYS & consumed)
        self.assertEqual(leaked, [],
                         "Known-unused set contains consumed keys: %s" % leaked)

        ghost = sorted(self._KNOWN_UNUSED_GRID_KEYS - grid_names)
        self.assertEqual(ghost, [],
                         "Known-unused set references keys absent from config.py: %s" % ghost)


if __name__ == "__main__":
    unittest.main()
