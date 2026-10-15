import unittest
from types import SimpleNamespace

from c4.run_llama_pretraining import (
    build_quality_result,
    parse_args,
    require_training_budget,
    set_model_pad_token_id,
    wandb_run_url,
)


class C4ModelSetupTests(unittest.TestCase):

    def test_quality_parser_accepts_explicit_offline_tokenizer_and_contract(self):
        args = parse_args([
            "--model_config", "model.json",
            "--batch_size", "1",
            "--tokenizer_path", "/fixed/tokenizer",
            "--contract_hash", "abc123",
            "--optimizer", "adamw",
        ])
        self.assertEqual(args.tokenizer_path, "/fixed/tokenizer")
        self.assertEqual(args.contract_hash, "abc123")

    def test_quality_result_records_formal_identity(self):
        result = build_quality_result(
            final_eval_loss=2.0,
            final_eval_tokens=100,
            best_eval_loss=1.9,
            best_eval_step=8,
            update_step=10,
            args=SimpleNamespace(
                dtype="bfloat16", optimizer="muon", contract_hash="contract"
            ),
            world_size=4,
            wandb_link=None,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["dtype"], "bfloat16")
        self.assertEqual(result["optimizer"], "muon")
        self.assertEqual(result["world_size"], 4)
        self.assertEqual(result["contract_hash"], "contract")

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
