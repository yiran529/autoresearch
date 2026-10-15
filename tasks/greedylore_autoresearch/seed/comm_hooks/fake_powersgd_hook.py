# mypy: allow-untyped-defs
import logging
from functools import partial
from typing import Dict, List
from collections import defaultdict

import torch
import torch.distributed as dist
import comm_hooks.default_hooks as default_hooks

from comm_hooks.utils import dtype_bits, tensor_bits, HookState, _get_allgather_out_list, check_error_identity

logger = logging.getLogger(__name__)


class FakePowerSGDState(HookState):
    
    def __init__(self,
        process_group: dist.ProcessGroup,
        matrix_approximation_rank: int = 8,
        update_proj_gap: int = 200,
        start_compress_iter: int = 10,
        min_compression_rate: int = 2.0,
        use_error_feedback: str = "ef14",
        batch_tensors_with_same_shape: bool = True,
        random: bool = False,
        uncompressed_names: List[str] = ["classifier", "embedding"],
        identity_proj: bool = False,
        scale: float = 1.0,
        random_seed: int = 0,
        sigma_type: int = 0,
        beta_ef: float = 0,
        error_inherit: int = 0,
    ):
        super().__init__(process_group)
        
        self.total_bit_before_compression = 0
        self.total_bit_after_compression = 0
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0

        self.iter = 0
        self.start_compress_iter = start_compress_iter
        self.compressor_name = f"PowerSGD"

        # random state
        self.random = random
        self.rng = torch.Generator()
        self.rng.manual_seed(random_seed)

        # EF-PowerSGD parameters
        self.matrix_approximation_rank = matrix_approximation_rank
        self.update_proj_gap = update_proj_gap
        self.min_compression_rate = min_compression_rate
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape
        self.update_tracker_every_gap = False
        self.svd_on_cpu = False
        self.uncompressed_names = uncompressed_names
        self.identity_proj = identity_proj
        self.scale = scale

        # Error feedback
        self.use_error_feedback = use_error_feedback
        self.checking = False

        # state dict
        self.error_dict: Dict[int, torch.Tensor] = {}       # store 1D error tensor for EF14 and EF21
        self.generator: Dict[int, torch.Generator] = {}       # store 1D error tensor for EF14 and EF21
        self.global_error_dict: Dict[int, torch.Tensor] = {}    # store 1D error tensor for EF21
        self.p_memory_dict: Dict[int, torch.Tensor] = {}
        self.q_memory_dict: Dict[int, torch.Tensor] = {}
        self.ps_dict: Dict[int, List[torch.Tensor]] = {}
        self.qs_dict: Dict[int, List[torch.Tensor]] = {}

        self.sigma_type = sigma_type
        self.beta_ef = beta_ef
        self.error_inherit = error_inherit

        # print all param of state
        print(f"state: {self.__dict__}")

def check_index_identity(bucket):
    input_tensor = bucket.buffer()
    tensors = bucket.gradients()
    idx = 0
    for tensor in tensors:
        assert torch.allclose(input_tensor[idx : idx + tensor.numel()], tensor.view(-1)), f"Tensor {idx} is not the same, {input_tensor[idx : idx + tensor.numel()]} != {tensor.view(-1)}"
        idx += tensor.numel()

def check_generator_identity(state: FakePowerSGDState, generator: torch.Generator):
    # print(f"Rank[{dist.get_rank()}] Iter[{state.iter}], generator: {generator}")
    device = generator.device
    process_group = state.process_group

    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    v = torch.rand(5, device=device, generator=generator)
    vl = [torch.ones_like(v) for _ in range(world_size)]
    dist.all_gather(vl, v, group=group_to_use)
    if dist.get_rank() == 0:
        for i, vi in enumerate(vl):
            assert vi.equal(vl[0]), f'diff !!!! idx{i}({vi}:idx{0}({vl[0]}))'

def check_shape_to_tensor_identity(shape_to_tensor1, shape_to_tensor2):
    assert len(shape_to_tensor1) == len(shape_to_tensor2), "The number of shapes should be equal."
    for shape, tensors1 in shape_to_tensor1.items():
        tensors2 = shape_to_tensor2[shape]
        assert len(tensors1) == len(tensors2), "The number of tensors should be equal."

def _should_compress_PowerSGD(
    num_rows, num_cols, matrix_approximation_rank, min_compression_rate
):
    """
    Recommend if tensor given is worth compressing.

    Returns a recommendation as to whether the 2D tensor described by the arguments is worth compressing,
    including statistics describing the expected savings from compression.  We consider a tensor worth
    compressing when ``min_compression_rate`` < uncompressed size / compressed size, where
    uncompressed size = ``num_rows`` * ``num_cols``,
    and compressed size = (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.

    The result of this function is a tuple of the form (compression_recommendation, uncompressed_el_count, compressed_el_count), where:

    compression_recommendation is true if the tensor is worth compressing, and false otherwise (see above);

    uncompressed_el_count is the uncompressed element count, i.e. ``num_rows`` * ``num_cols``; and,

    compress_el_count is the element count after compression, i.e. (``num_rows`` + ``num_cols``) * ``matrix_approximation_rank``.
    """  # noqa: B950
    uncompressed_size = num_rows * num_cols
    
    compressed_size = matrix_approximation_rank * (num_rows + num_cols)
    return (
        compressed_size * min_compression_rate < uncompressed_size,
        uncompressed_size,
        compressed_size,
    )

def gram_schmidt_inplace(A):
    """
    Perform Gram-Schmidt orthogonalization on A in-place.
    A: (n, r) tensor where n > r.
    """
    n, r = A.shape
    for i in range(r):
        # Normalize the i-th column
        col_norm = torch.norm(A[:, i], p=2)  # L2 norm
        A[:, i].div_(col_norm)  # In-place normalization
        
        # Orthogonalize the remaining columns
        if i < r - 1:
            # Compute the projection of A[:, i+1:] onto A[:, i]
            projections = torch.mm(A[:, i:i+1].T, A[:, i+1:])  # (1, r-i-1)
            # Subtract the projections in-place
            A[:, i+1:].sub_(torch.mm(A[:, i:i+1], projections))  # (n, r-i-1)
    return A

def divide_tensors_to_compress_PowerSGD_single(bucket: dist.GradBucket, state: FakePowerSGDState):
    uncompressed_tensors = []
    compressed_tensors = []
    total_Ps_size, total_Qs_size = 0, 0
    tensors, params = bucket.gradients(), bucket.parameters()
    for tensor, param in zip(tensors, params):
        # TODO: add branch when n is much larger than m
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        should_compress, uncompressed_size, compressed_size = _should_compress_PowerSGD(
            n, m, matrix_approximation_rank, state.min_compression_rate
        )
        # Don't compress if embedding in name or classifier in name
        if any(name in state.param_to_name[param] for name in state.uncompressed_names):
            if state.iter == state.start_compress_iter and dist.get_rank() == 0:
                print(f"Skip compressing {state.param_to_name[param]}")
            should_compress = False
        state.total_numel_before_compression += uncompressed_size
        state.total_bit_before_compression += uncompressed_size * dtype_bits(tensor)
        if should_compress:
            compressed_tensors.append(tensor)
            total_Ps_size += n * matrix_approximation_rank
            total_Qs_size += m * matrix_approximation_rank
            state.total_numel_after_compression += compressed_size
            state.total_bit_after_compression += compressed_size * dtype_bits(tensor)
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += uncompressed_size
            state.total_bit_after_compression += uncompressed_size * dtype_bits(tensor)
    return uncompressed_tensors, compressed_tensors, total_Ps_size, total_Qs_size

def fake_PowerSGD_hook(
    state: FakePowerSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    if state.use_error_feedback == "ef14":
        return fake_PowerSGD_hook_ef14(state, bucket)
    else:
        raise ValueError(f"Unsupported error feedback method: {state.use_error_feedback}")

def fake_PowerSGD_hook_ef14(
    state: FakePowerSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor; Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    input_tensor = bucket.buffer()
    tensors = bucket.gradients()
    bucket_index = bucket.index()
    total_length = input_tensor.shape[0]

    # if dist.get_rank() == 0:
    #     print(f"Rank[{dist.get_rank()}] Iter[{state.iter}], bucket index[{bucket_index}], total length: {total_length}")

    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor)
    
    # Apply PowerSGD after `start_compress_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    if bucket_index not in state.generator:
        state.generator[bucket_index] = torch.Generator(device = device)
        state.generator[bucket_index].manual_seed(torch.randint(0, 100000, (1,), generator=state.rng).item())


    # Step O: compute the difference between the input tensor and the local error tensor
    if bucket_index not in state.error_dict:
        state.error_dict[bucket_index] = torch.zeros_like(input_tensor)
    input_tensor.add_(state.error_dict[bucket_index], alpha=1.0)   # input_tensor = \nabla F_i + E_{i-1}
    input_tensor_cp = input_tensor.detach().clone()

    # Step 1: Initialize the memory for storing Ps and Qs
    uncompressed_tensors, compressed_tensors, total_Ps_size, total_Qs_size  = divide_tensors_to_compress_PowerSGD_single(bucket, state)

    if bucket_index not in state.ps_dict:
        state.p_memory_dict[bucket_index] = torch.empty(total_Ps_size, dtype=dtype, device=device)
        state.q_memory_dict[bucket_index] = torch.randn(total_Qs_size, dtype=dtype, device=device, generator=state.generator[bucket_index])
        state.ps_dict[bucket_index] = []
        state.qs_dict[bucket_index] = []
        p_index, q_index = 0, 0
        for tensor in compressed_tensors:
            n, m = tensor.shape
            matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
            state.ps_dict[bucket_index].append(state.p_memory_dict[bucket_index][p_index : p_index + n * matrix_approximation_rank].view(n, matrix_approximation_rank))
            state.qs_dict[bucket_index].append(state.q_memory_dict[bucket_index][q_index : q_index + m * matrix_approximation_rank].view(m, matrix_approximation_rank))
            p_index += n * matrix_approximation_rank
            q_index += m * matrix_approximation_rank
        assert p_index == total_Ps_size
        assert q_index == total_Qs_size

    # Step 2: Simulate the compression and decompression process
    
    # 2.1 allreduce uncompressed tensors
    # dist.all_reduce(input_tensor, group=group_to_use)
    # input_tensor.div_(world_size)

    # 2.2 simulate the compression and decompression process
    # P = GQ
    for tensor, P, Q in zip(compressed_tensors, state.ps_dict[bucket_index], state.qs_dict[bucket_index]):
        gram_schmidt_inplace(Q)
        torch.mm(tensor, Q, out=P)
    # P = allreduce(P)
    dist.all_reduce(state.p_memory_dict[bucket_index], group=group_to_use)
    state.p_memory_dict[bucket_index].div_(world_size)
    # P = orth(P), Q = G^T P
    for tensor, P, Q in zip(compressed_tensors, state.ps_dict[bucket_index], state.qs_dict[bucket_index]):
        gram_schmidt_inplace(P)
        torch.mm(tensor.T, P, out=Q)
    # Q = allreduce(Q)
    dist.all_reduce(state.q_memory_dict[bucket_index], group=group_to_use)
    state.q_memory_dict[bucket_index].div_(world_size)
    # G = PQ^T
    for tensor, P, Q in zip(compressed_tensors, state.ps_dict[bucket_index], state.qs_dict[bucket_index]):
        torch.mm(P, Q.T, out=tensor)
                
    # Step 4: compute the difference between the compressed tensor and the input tensor
    # iter, bucket index, error tensor
    # if dist.get_rank() == 0:
    #     print(f"Rank[{dist.get_rank()}] Iter[{state.iter}], bucket index[{bucket_index}], compress error: {torch.norm(input_tensor_cp - input_tensor).item()}")
    state.error_dict[bucket_index].lerp_(input_tensor_cp - input_tensor, 1 - state.beta_ef)

    # Step 3: Handle uncompressed tensors
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )
    dist.all_reduce(uncompressed_tensors_memory, group=group_to_use)
    uncompressed_tensors_memory.div_(world_size)
    idx = 0
    for tensor in uncompressed_tensors:
        tensor.copy_(uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor))
        idx += tensor.numel()

    # Step 4.5: Scaling low rank part
    for tensor in compressed_tensors:
        tensor.mul_(state.scale)

    # Step 5: increase the iteration counter
    state.maybe_increase_iter(bucket)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)
    return fut
   