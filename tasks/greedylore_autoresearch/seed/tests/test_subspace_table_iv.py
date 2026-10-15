import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

from comm_hooks.subspace_hook import (
    SubspaceState,
    _should_compress_subspace,
    estimate_subspace_scores,
    subspace_hook,
)


class _Bucket:
    def __init__(self, parameter):
        self.parameter = parameter
        self.tensor = torch.diag(torch.tensor([1., 20., 2., 3.]))

    def buffer(self):
        return self.tensor.view(-1)

    def gradients(self):
        return [self.tensor]

    def parameters(self):
        return [self.parameter]

    def index(self):
        return 0

    def is_last(self):
        return True


class SubspaceTableIVTests(unittest.TestCase):
    def test_second_two_dimensional_step_compresses(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo", init_method=f"file://{directory}/init", rank=0, world_size=1
            )
            try:
                parameter = torch.nn.Parameter(torch.zeros(4, 4))
                bucket = _Bucket(parameter)
                state = SubspaceState(
                    dist.group.WORLD, matrix_approximation_rank=1,
                    start_compress_iter=0, update_proj_gap=200,
                    use_error_feedback="ef14", uncompressed_names=[],
                )
                state.param_to_name = {parameter: "weight"}
                subspace_hook(state, bucket).wait()
                with patch("comm_hooks.subspace_hook.torch.cuda.is_available", return_value=False):
                    second = subspace_hook(state, bucket).wait()
                self.assertEqual(second.shape, (16,))
                self.assertEqual(state.iter, 2)
            finally:
                dist.destroy_process_group()

    def test_rank_128_is_eligible_at_rate_one(self):
        self.assertFalse(_should_compress_subspace(256, 256, 128, 2.0)[0])
        self.assertTrue(_should_compress_subspace(256, 256, 128, 1.0)[0])

    def test_scores_select_dominant_axis(self):
        basis = torch.eye(3).unsqueeze(0)
        gradient = torch.diag(torch.tensor([1., 100., 2.])).unsqueeze(0)
        scores = estimate_subspace_scores(
            [basis], [gradient], torch.Generator().manual_seed(1), None
        )
        self.assertEqual(int(scores[0].argmax()), 1)


if __name__ == "__main__":
    unittest.main()
