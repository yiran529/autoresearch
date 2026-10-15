import subprocess
import sys
import unittest


class GlueCliTests(unittest.TestCase):
    def test_help_starts_with_installed_transformers(self):
        result = subprocess.run(
            [sys.executable, "glue/run_glue_no_trainer_HF.py", "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--local_glue_cache_root", result.stdout)


if __name__ == "__main__":
    unittest.main()
