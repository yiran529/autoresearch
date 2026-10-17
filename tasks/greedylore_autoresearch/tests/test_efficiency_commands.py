from greedylore_autoresearch_grader.efficiency import (
    build_timing_command,
    iter_timing_runs,
)


def test_timing_error_inherit_is_accepted_by_integer_parser(contract, tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--error_inherit', type=int)
    for cell in contract.efficiency.cells.values():
        spec = build_timing_command(cell, 0, tmp_path, tmp_path / 'out', (0, 1, 2, 3))
        args, _ = parser.parse_known_args(spec.argv)
        assert args.error_inherit == 0


def has_args(command, *expected):
    size = len(expected)
    return any(tuple(command[index:index + size]) == expected for index in range(len(command)))


def test_efficiency_matrix_is_only_130m_adamw(contract, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    commands = [
        build_timing_command(cell, 0, candidate, tmp_path / name, (0, 2, 4, 6)).argv
        for name, cell in contract.efficiency.cells.items()
    ]

    assert len(commands) == 2
    assert all(has_args(command, "--optimizer", "adamw", "--lr", "0.0025") for command in commands)
    assert all(has_args(command, "--batch_size", "1") for command in commands)
    assert all(has_args(command, "--warmup_iterations", "101") for command in commands)
    assert all(has_args(command, "--measured_iterations", "400") for command in commands)
    assert all(has_args(command, "--record_bucket_layout") for command in commands)
    assert all("--strict_blocking_communication" not in command for command in commands)


def test_efficiency_repetitions_use_fixed_reversed_order_and_shm(contract, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    specs = list(iter_timing_runs(contract.efficiency, candidate, tmp_path / "out", (1, 3, 5, 7)))

    assert [(run.repetition, run.cell_name) for run in specs] == [
        (0, "130m_adamw_bf16"),
        (0, "130m_adamw_fp32"),
        (1, "130m_adamw_fp32"),
        (1, "130m_adamw_bf16"),
    ]
    assert all(run.spec.env["NCCL_SHM_DISABLE"] == "0" for run in specs)
    assert len({run.spec.log_dir for run in specs}) == 4
    assert all(run.spec.timeout == 2 * 60 * 60 for run in specs)
