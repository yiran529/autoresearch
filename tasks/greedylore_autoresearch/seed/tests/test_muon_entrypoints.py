import ast
import argparse
import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


class TestMuonEntrypoints(unittest.TestCase):
    def assert_script_help_exposes_muon(self, relative_path, *, allocate_tty=False):
        command = [str(PYTHON), str(ROOT / relative_path), "--help"]
        if allocate_tty:
            command = ["script", "-qec", " ".join(command), "/dev/null"]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--muon_mu", completed.stdout)

    def assert_parse_function_help_exposes_muon(self, relative_path):
        from optimizer import add_muon_args

        path = ROOT / relative_path
        tree = ast.parse(path.read_text())
        parse_function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
        )
        module = ast.Module(body=[parse_function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "argparse": argparse,
            "task_to_keys": {
                "cola": ("sentence", None),
                "mnli": ("premise", "hypothesis"),
                "mrpc": ("sentence1", "sentence2"),
                "qnli": ("question", "sentence"),
                "qqp": ("question1", "question2"),
                "rte": ("sentence1", "sentence2"),
                "sst2": ("sentence", None),
                "stsb": ("sentence1", "sentence2"),
                "wnli": ("sentence1", "sentence2"),
            },
            "SchedulerType": str,
            "add_muon_args": add_muon_args,
        }
        exec(compile(module, str(path), "exec"), namespace)
        old_argv = sys.argv
        output = io.StringIO()
        try:
            sys.argv = [str(path), "--help"]
            with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
                namespace["parse_args"]()
            self.assertEqual(raised.exception.code, 0)
        finally:
            sys.argv = old_argv
        self.assertIn("--muon_mu", output.getvalue())

    def test_c4_exposes_muon_arguments(self):
        self.assert_script_help_exposes_muon("c4/run_llama_pretraining.py")

    def test_glue_exposes_muon_arguments(self):
        self.assert_parse_function_help_exposes_muon("glue/run_glue_no_trainer_HF.py")

    def test_cifar_exposes_muon_arguments(self):
        self.assert_script_help_exposes_muon("pytorch-cifar/main.py", allocate_tty=True)


if __name__ == "__main__":
    unittest.main()
