import json
import tempfile
import unittest
from pathlib import Path

from c4.pept_utils.c4_data import load_c4_split


class LocalC4Tests(unittest.TestCase):
    def test_parent_directory_finds_english_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            english = Path(directory) / "en-30-shards" / "en"
            english.mkdir(parents=True)
            for split, count in (("train", "01024"), ("validation", "00008")):
                path = english / f"c4-{split}.00000-of-{count}.json"
                path.write_text(json.dumps({"text": split, "timestamp": "", "url": ""}) + "\n")
            self.assertEqual(next(iter(load_c4_split(directory, "train")))["text"], "train")
            self.assertEqual(next(iter(load_c4_split(directory, "validation")))["text"], "validation")


if __name__ == "__main__":
    unittest.main()
