from greedylore_autoresearch_grader.quality import build_quality_command


def test_quality_error_inherit_is_accepted_by_integer_parser(contract, tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--error_inherit', type=int)
    for profile in contract.quality.values():
        spec = build_quality_command(profile, tmp_path, tmp_path / 'out', (0, 1, 2, 3))
        args, _ = parser.parse_known_args(spec.argv)
        assert args.error_inherit == 0


def has_args(command, *expected):
    size = len(expected)
    return any(tuple(command[index:index + size]) == expected for index in range(len(command)))


def test_130m_quality_preserves_full_schedule_prefix(contract, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    spec = build_quality_command(
        contract.quality["llama130m_muon"], candidate, tmp_path / "out", (2, 4, 5, 7)
    )

    assert has_args(spec.argv, "--num_training_steps", "20000")
    assert has_args(spec.argv, "--stop_after_steps", "2500")
    assert has_args(spec.argv, "--optimizer", "muon", "--lr", "0.01")
    assert has_args(spec.argv, "--muon_scalar_lr", "0.001")
    assert has_args(spec.argv, "--tokenizer_path", str(contract.data.tokenizer_path))
    assert has_args(spec.argv, "--activation_checkpointing")
    assert spec.env["CUDA_VISIBLE_DEVICES"] == "2,4,5,7"
    assert spec.env["WANDB_MODE"] == "disabled"
    assert spec.cwd == candidate
    assert spec.timeout == 24 * 60 * 60


def test_60m_exploration_stops_at_4000_with_original_schedule(contract, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    spec = build_quality_command(
        contract.quality["llama60m_adamw"], candidate, tmp_path / "out", (0, 1, 2, 3)
    )

    assert has_args(spec.argv, "--num_training_steps", "8393")
    assert has_args(spec.argv, "--stop_after_steps", "4000")
    assert has_args(spec.argv, "--warmup_steps", "1000")
    assert has_args(spec.argv, "--optimizer", "adamw", "--lr", "0.0025")
    assert has_args(spec.argv, "--batch_size", "128")
    assert has_args(spec.argv, "--dtype", "bfloat16")
