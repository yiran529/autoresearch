"""Read prepared GLUE Arrow splits from the local Hugging Face cache."""

from pathlib import Path

from datasets import Dataset, DatasetDict


def load_local_glue(cache_root: str | Path, task_name: str) -> DatasetDict:
    task_dir = Path(cache_root) / task_name
    splits = ["train", "validation"]
    if task_name == "mnli":
        splits = ["train", "validation_matched", "validation_mismatched"]

    candidates = sorted(task_dir.glob("*/*/glue-train.arrow"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one prepared GLUE cache for {task_name} under {task_dir}; found {len(candidates)}")
    data_dir = candidates[0].parent
    result = DatasetDict()
    for split in splits:
        path = data_dir / f"glue-{split}.arrow"
        if not path.is_file():
            raise FileNotFoundError(f"Missing GLUE {task_name} split: {path}")
        result[split] = Dataset.from_file(str(path))
    return result
