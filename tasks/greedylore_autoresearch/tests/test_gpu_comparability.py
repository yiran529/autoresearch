from greedylore_autoresearch_grader.gpu import (
    EfficiencyEnvironment,
    gpu_topology_fingerprint,
    validate_efficiency_comparability,
)


def env(uuids=("gpu-a", "gpu-b", "gpu-c", "gpu-d"), topo="links"):
    return EfficiencyEnvironment(uuids, ("A100",) * 4, topo, (0, 0, 1, 1), {"NCCL_ALGO": "Ring"})


def test_efficiency_accepts_same_gpu_set():
    assert validate_efficiency_comparability(env(), env()).valid


def test_efficiency_warns_on_different_topology():
    candidate = env(("gpu-a", "gpu-c", "gpu-d", "gpu-f"), "different-links")
    result = validate_efficiency_comparability(env(), candidate)
    assert result.valid
    assert result.status == "cross_topology"
    assert "topology_fingerprint" in result.differences
    assert result.warnings


def test_topology_fingerprint_is_order_stable():
    one = gpu_topology_fingerprint(("b", "a"), "matrix", {"NCCL_ALGO": "Ring"}, (1, 0))
    two = gpu_topology_fingerprint(("a", "b"), "matrix", {"NCCL_ALGO": "Ring"}, (0, 1))
    assert one == two


def test_different_uuid_set_still_requires_matching_nccl():
    dense = env()
    equivalent = env(("gpu-e", "gpu-f", "gpu-g", "gpu-h"))
    assert validate_efficiency_comparability(dense, equivalent).valid
    mismatched = EfficiencyEnvironment(
        equivalent.gpu_uuids,
        equivalent.gpu_names,
        equivalent.topology_fingerprint,
        equivalent.numa_nodes,
        {"NCCL_ALGO": "Tree"},
    )
    assert not validate_efficiency_comparability(dense, mismatched).valid


def test_same_uuid_set_still_requires_unique_cards_and_matching_environment():
    dense = env()
    duplicate = env(("gpu-a", "gpu-a", "gpu-c", "gpu-d"))
    assert not validate_efficiency_comparability(dense, duplicate).valid
    changed_nccl = EfficiencyEnvironment(
        dense.gpu_uuids,
        dense.gpu_names,
        dense.topology_fingerprint,
        dense.numa_nodes,
        {"NCCL_ALGO": "Tree"},
    )
    assert not validate_efficiency_comparability(dense, changed_nccl).valid


def test_numa_differences_and_unknown_numa_warn_without_rejecting():
    from dataclasses import replace

    for numa in ((2, 2, 3, 3), (None,) * 4):
        result = validate_efficiency_comparability(env(), replace(env(), numa_nodes=numa))
        assert result.valid
        assert result.status == 'cross_topology'
        assert result.warnings


def test_gpu_model_and_count_still_reject():
    from dataclasses import replace

    for candidate in (replace(env(), gpu_names=('H100',) * 4),
                      replace(env(), gpu_uuids=('a', 'b', 'c'))):
        result = validate_efficiency_comparability(env(), candidate)
        assert not result.valid
        assert result.reason == 'incomparable_hardware'
