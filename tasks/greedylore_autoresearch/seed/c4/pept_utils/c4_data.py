"""Load C4 English shards from a local directory or a datasets identifier."""

from pathlib import Path

import datasets


def load_c4_split(dataset_path: str, split: str):
    if split not in ("train", "validation"):
        raise ValueError(f"unsupported C4 split: {split}")

    root = Path(dataset_path)
    if root.is_dir():
        count = "01024" if split == "train" else "00008"
        name = f"c4-{split}.[0-9][0-9][0-9][0-9][0-9]-of-{count}.json*"
        files = sorted((root / "en").glob(name))
        if not files:
            files = sorted((root / "en-30-shards" / "en").glob(name))
        if not files:
            raise FileNotFoundError(f"No C4 {split} shards under {root}")
        return datasets.load_dataset(
            "json", data_files={split: [str(path) for path in files]},
            split=split, streaming=True,
        )

    if dataset_path == "allenai/c4":
        return datasets.load_dataset(dataset_path, "en", split=split, streaming=True)
    return datasets.load_dataset(dataset_path, split=split, streaming=True)
