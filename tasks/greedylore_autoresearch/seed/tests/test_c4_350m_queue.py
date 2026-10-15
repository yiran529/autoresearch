import json
from pathlib import Path
import tempfile
import unittest

from c4.run_llama_pretraining import training_target_steps, training_batches
from c4.scripts.queue_table_iv_350m_muon import choose_lr, stages, require_unused_numbers
from c4.scripts.queue_table_iv_350m_muon import require_idle


class TableIV350MTests(unittest.TestCase):
    def test_recent_utilization_can_settle_before_next_stage(self):
        from unittest.mock import patch
        busy = '0, 0, 100\n1, 0, 100\n2, 0, 100\n3, 0, 100\n'
        idle = '0, 0, 0\n1, 0, 0\n2, 0, 0\n3, 0, 0\n'
        with patch('c4.scripts.queue_table_iv_350m_muon.subprocess.check_output',
                   side_effect=[busy, idle]), \
             patch('time.sleep'):
            require_idle('0,1,2,3')

    def test_existing_data_repeats_only_when_requested(self):
        from itertools import islice
        self.assertEqual(list(training_batches([1, 2], False)), [1, 2])
        self.assertEqual(list(islice(training_batches([1, 2], True), 5)), [1, 2, 1, 2, 1])
        with self.assertRaises(RuntimeError):
            next(training_batches([], True))

    def test_repeated_runs_keep_full_batches_at_data_boundaries(self):
        from itertools import islice
        import torch
        batches = [{'input_ids': torch.tensor([[1], [2]])},
                   {'input_ids': torch.tensor([[3]])}]
        output = list(islice(training_batches(batches, True, batch_size=2), 3))
        self.assertEqual([batch['input_ids'].tolist() for batch in output],
                         [[[1], [2]], [[1], [2]], [[1], [2]]])

    def test_queue_log_does_not_conflict_but_existing_run_does(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'CM248-CM252-table-iv-350m-muon.queue.log').touch()
            require_unused_numbers(root)
            (root / 'CM248-m001-dense-old').mkdir()
            with self.assertRaises(FileExistsError):
                require_unused_numbers(root)

    def test_sweep_stops_without_changing_scheduler_horizon(self):
        self.assertEqual(training_target_steps(60000, 10000), 10000)
        self.assertEqual(training_target_steps(60000, None), 60000)
        for stop in (0, -1, 60001):
            with self.assertRaises(ValueError):
                training_target_steps(60000, stop)

    def test_selection_uses_endpoint_loss_and_rejects_incomplete_or_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = [('a', '0.01'), ('b', '0.005')]
            for run, loss in [('a', 3.2), ('b', 3.1)]:
                (root / run).mkdir()
                (root / run / 'all_results.json').write_text(json.dumps(
                    dict(update_step=10000, final_eval_loss=loss)))
            self.assertEqual(choose_lr(root, runs), '0.005')
            for step, loss in [(9999, 3.0), (10000, float('nan'))]:
                (root / 'b' / 'all_results.json').write_text(json.dumps(
                    dict(update_step=step, final_eval_loss=loss)))
                with self.assertRaises(ValueError):
                    choose_lr(root, runs)

    def test_formal_pair_has_disjoint_gpus_then_r256(self):
        sweep, pair, last = stages('0.005')
        self.assertEqual([cell['steps'] for cell in sweep], [10000, 10000])
        self.assertEqual([cell['arm'] for cell in pair], ['dense', 'r32'])
        self.assertEqual([cell['gpus'] for cell in pair], ['0,1,2,3', '4,5,6,7'])
        self.assertEqual([cell['steps'] for cell in pair], [60000, 60000])
        self.assertEqual([(cell['arm'], cell['steps'], cell['lr']) for cell in last],
                         [('r256', 60000, '0.005')])


if __name__ == '__main__':
    unittest.main()
