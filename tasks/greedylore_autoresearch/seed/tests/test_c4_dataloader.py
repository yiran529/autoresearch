import unittest

import datasets
import datasets.distributed
import torch

from c4.pept_utils.dataloader import PreprocessedIterableDataset


class NumberTokenizer:
    def __call__(self, text, **kwargs):
        value = int(text)
        return {
            "input_ids": torch.tensor([[value]]),
            "attention_mask": torch.tensor([[1]]),
        }


class C4DataloaderTests(unittest.TestCase):
    def test_hf_worker_shards_are_not_split_twice(self):
        data = datasets.Dataset.from_dict(
            {"text": [str(i) for i in range(16)]}
        ).to_iterable_dataset(num_shards=4)
        wrapped = PreprocessedIterableDataset(data, NumberTokenizer(), 1, 1)
        loader = torch.utils.data.DataLoader(wrapped, batch_size=None, num_workers=4)

        observed = [int(batch["input_ids"].item()) for batch in loader]

        self.assertEqual(sorted(observed), list(range(16)))

    def test_thirty_shards_across_four_ranks_and_workers(self):
        data = datasets.Dataset.from_dict(
            {"text": [str(i) for i in range(120)]}
        ).to_iterable_dataset(num_shards=30)
        observed = []
        for rank in range(4):
            rank_data = datasets.distributed.split_dataset_by_node(
                data, rank=rank, world_size=4
            )
            wrapped = PreprocessedIterableDataset(rank_data, NumberTokenizer(), 1, 1)
            loader = torch.utils.data.DataLoader(wrapped, batch_size=None, num_workers=4)
            observed.extend(int(batch["input_ids"].item()) for batch in loader)

        self.assertEqual(sorted(observed), list(range(120)))


if __name__ == "__main__":
    unittest.main()
