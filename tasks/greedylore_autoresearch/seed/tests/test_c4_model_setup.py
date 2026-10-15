import unittest
from types import SimpleNamespace

from c4.run_llama_pretraining import (
    require_training_budget,
    set_model_pad_token_id,
    wandb_run_url,
)


class C4ModelSetupTests(unittest.TestCase):
    def test_model_without_generation_config_accepts_tokenizer_pad_id(self):
        model = SimpleNamespace(
            module=SimpleNamespace(
                config=SimpleNamespace(pad_token_id=-1),
                generation_config=None,
            )
        )
        set_model_pad_token_id(model, 0)
        self.assertEqual(model.module.config.pad_token_id, 0)

    def test_short_dataset_cannot_be_reported_as_completed_training(self):
        with self.assertRaisesRegex(RuntimeError, "5220 of 10000"):
            require_training_budget(5220, 10000)
        require_training_budget(10000, 10000)

    def test_current_wandb_url_property_is_used(self):
        self.assertEqual(
            wandb_run_url(SimpleNamespace(url="https://wandb.ai/example/run")),
            "https://wandb.ai/example/run",
        )


if __name__ == "__main__":
    unittest.main()
