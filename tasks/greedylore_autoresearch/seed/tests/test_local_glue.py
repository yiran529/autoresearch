import shutil
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset

from glue.local_glue import load_local_glue


class LocalGlueTests(unittest.TestCase):
    def test_loads_train_and_validation_from_arrow_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cola" / "0.0.0" / "revision"
            cache.mkdir(parents=True)
            for split, sentence in (("train", "training"), ("validation", "evaluation")):
                saved = root / f"saved-{split}"
                Dataset.from_dict({"sentence": [sentence], "label": [1]}).save_to_disk(saved)
                shutil.copyfile(next(saved.glob("*.arrow")), cache / f"glue-{split}.arrow")

            result = load_local_glue(root, "cola")

            self.assertEqual(result["train"][0]["sentence"], "training")
            self.assertEqual(result["validation"][0]["sentence"], "evaluation")

    def test_loads_both_mnli_validation_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "mnli" / "0.0.0" / "revision"
            cache.mkdir(parents=True)
            for split in ("train", "validation_matched", "validation_mismatched"):
                saved = root / f"saved-{split}"
                Dataset.from_dict({"premise": [split], "hypothesis": ["text"], "label": [1]}).save_to_disk(saved)
                shutil.copyfile(next(saved.glob("*.arrow")), cache / f"glue-{split}.arrow")

            result = load_local_glue(root, "mnli")

            self.assertEqual(result["validation_matched"][0]["premise"], "validation_matched")
            self.assertEqual(result["validation_mismatched"][0]["premise"], "validation_mismatched")


if __name__ == "__main__":
    unittest.main()
