import json
import tempfile
import unittest
from pathlib import Path

from c4.scripts.queue_table_iv_130m_muon_lr import choose_best_dense


class DenseMuonLRSweepTests(unittest.TestCase):
    def test_selects_lower_final_loss_after_both_complete_20000_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, loss in (("dense-0p01", 3.31), ("dense-0p005", 3.29)):
                path = root / run_id
                path.mkdir()
                (path / "all_results.json").write_text(
                    json.dumps({"update_step": 20000, "final_eval_loss": loss})
                )
            self.assertEqual(
                choose_best_dense(root, (("dense-0p01", "0.01"), ("dense-0p005", "0.005"))),
                "0.005",
            )

    def test_rejects_incomplete_run_instead_of_selecting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, step, loss in (("dense-0p01", 20000, 3.31),
                                       ("dense-0p005", 5000, 3.20)):
                path = root / run_id
                path.mkdir()
                (path / "all_results.json").write_text(
                    json.dumps({"update_step": step, "final_eval_loss": loss})
                )
            with self.assertRaises(ValueError):
                choose_best_dense(root, (("dense-0p01", "0.01"), ("dense-0p005", "0.005")))


if __name__ == "__main__":
    unittest.main()
