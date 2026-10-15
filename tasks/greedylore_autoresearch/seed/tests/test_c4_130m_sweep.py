import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

from c4.scripts.sweep_table_iv_130m_lr import (
    idle_gpu_ids,
    read_eval_loss,
    run_screened,
    select_stable_gpus,
)


ROOT = Path(__file__).resolve().parents[1]


class C4130MSweepTests(unittest.TestCase):
    def test_selects_same_four_idle_gpus_in_consecutive_checks(self):
        first = idle_gpu_ids("0, 0, 0\n1, 0, 0\n2, 0, 0\n3, 0, 0\n4, 4000, 5\n5, 5000, 95\n")
        second = idle_gpu_ids("0, 0, 0\n1, 0, 0\n2, 0, 0\n3, 2000, 0\n4, 0, 0\n5, 0, 0\n")
        self.assertIsNone(select_stable_gpus(first, second))
        third = idle_gpu_ids("0, 0, 0\n1, 0, 0\n2, 0, 0\n3, 0, 0\n4, 0, 0\n5, 0, 0\n")
        self.assertEqual(select_stable_gpus(second, third), (0, 1, 2, 4))

    def test_reads_only_requested_evaluation_step(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text("Eval loss at step 4000: 3.1\nEval loss at step 5000: 3.74\n")
            self.assertEqual(read_eval_loss(log, 5000), 3.74)
            self.assertIsNone(read_eval_loss(log, 6000))

    def test_equal_loss_stops_only_the_candidate_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            command = ["python3", "-c", "import pathlib,time; "
                       f"pathlib.Path({str(log)!r}).write_text('Eval loss at step 5000: 3.74\\n'); "
                       "time.sleep(30)"]
            result = run_screened(command, log, baseline_loss=3.74, poll_seconds=0.05)
            self.assertEqual(result, "stopped_no_improvement")

    def test_lower_loss_allows_candidate_to_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            command = ["python3", "-c", "import pathlib; "
                       f"pathlib.Path({str(log)!r}).write_text('Eval loss at step 5000: 3.73999\\n')"]
            result = run_screened(command, log, baseline_loss=3.74, poll_seconds=0.05)
            self.assertEqual(result, "completed_improved")

    def test_nonfinite_loss_stops_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            command = ["python3", "-c", "import pathlib,time; "
                       f"pathlib.Path({str(log)!r}).write_text('Eval loss at step 5000: nan\\n'); "
                       "time.sleep(30)"]
            result = run_screened(command, log, baseline_loss=3.74, poll_seconds=0.05)
            self.assertEqual(result, "stopped_nonfinite")

    def test_r64_sweep_overrides_learning_rates_and_stop_only(self):
        script = ROOT / "c4/scripts/run_table_iv_130m.bash"
        base = shlex.split(subprocess.check_output(["bash", script, "r64", "--dry-run"], cwd=ROOT, text=True))
        env = os.environ.copy()
        env.update(MUON_MATRIX_LR="0.1", MUON_SCALAR_LR="0.0002", RUN_ID="CM039-sweep-test")
        sweep = shlex.split(subprocess.check_output(["bash", script, "r64", "--dry-run"], cwd=ROOT, env=env, text=True))
        for flag, value in (("--lr", "0.1"), ("--muon_scalar_lr", "0.0002"),
                            ("--compress_rank", "64"),
                            ("--num_training_steps", "20000"), ("--warmup_steps", "2000")):
            self.assertEqual(sweep[sweep.index(flag) + 1], value)
        self.assertNotIn("--stop_after_steps", sweep)
        self.assertEqual(sweep[sweep.index("--wandb_run_name") + 1], "CM039-sweep-test")


if __name__ == "__main__":
    unittest.main()
