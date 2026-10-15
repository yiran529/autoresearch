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


class FakeSubspaceState(HookState):
    
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
        self.compressor_name = f"Subspace"

        # random state
        self.random = random
        self.rng = torch.Generator()
        self.rng.manual_seed(random_seed)

        # EF-Subspace parameters
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
        self.m_memory_dict: Dict[int, torch.Tensor] = {}
        self.ms_dict: Dict[int, List[torch.Tensor]] = {}

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

def check_generator_identity(state: FakeSubspaceState, generator: torch.Generator):
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

def _should_compress_subspace(
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
    _max = max(num_rows, num_cols)
    compressed_size = _max * matrix_approximation_rank
    return (
        compressed_size * min_compression_rate < uncompressed_size,
        uncompressed_size,
        compressed_size,
    )

def divide_tensors_to_compress_subspace_single(bucket: dist.GradBucket, state: FakeSubspaceState):
    uncompressed_tensors = []
    compressed_tensors = []
    total_Ms_size, total_Rs_size = 0, 0
    tensors, params = bucket.gradients(), bucket.parameters()
    for tensor, param in zip(tensors, params):
        # TODO: add branch when n is much larger than m
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        _min, _max = min(n, m), max(n, m)
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        should_compress, uncompressed_size, compressed_size = _should_compress_subspace(
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
            total_Ms_size += _min * _min
            total_Rs_size += _max * matrix_approximation_rank
            state.total_numel_after_compression += compressed_size
            state.total_bit_after_compression += compressed_size * dtype_bits(tensor)
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += uncompressed_size
            state.total_bit_after_compression += uncompressed_size * dtype_bits(tensor)
    return uncompressed_tensors, compressed_tensors, total_Ms_size, total_Rs_size

def fake_subspace_hook(
    state: FakeSubspaceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    if state.use_error_feedback == "ef14":
        return fake_subspace_hook_ef14(state, bucket)
    else:
        raise ValueError(f"Unsupported error feedback method: {state.use_error_feedback}")

def fake_subspace_hook_ef14(
    state: FakeSubspaceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:


    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor; Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    input_tensor = bucket.buffer()
    tensors = bucket.gradients()
    bucket_index = bucket.index()
    total_length = input_tensor.shape[0]

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
    

    # Args for update
    update_projection = ((state.iter - state.start_compress_iter) % state.update_proj_gap == 0)
    
    # During the update_projection iteration, update the projection matrix.
    if update_projection:
        # 1. Set up local error tensor for EF14
        # E_i = 0
        if bucket_index in state.error_dict:
            if state.error_inherit == 1:
                input_tensor.add_(state.error_dict[bucket_index], alpha=1.0)   # input_tensor = \nabla F_i + E_{i-1}
                pass
            else:
                state.error_dict[bucket_index].zero_()
        else:
            print("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)

        # 2. Allreduce the difference to calculate the average.
        dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
        input_tensor.div_(world_size)

        # 3. Divide all the tensors into two groups, reate dict related to projection matrix
        uncompressed_tensors, compressed_tensors, total_Ms_size, total_Rs_size = divide_tensors_to_compress_subspace_single(bucket, state)

        if bucket_index not in state.m_memory_dict:
            state.m_memory_dict[bucket_index] = torch.empty(
                total_Ms_size, device=device, dtype=dtype
            )
        if bucket_index not in state.ms_dict:
            state.ms_dict[bucket_index] = list()

        # 4. Tensors have been allreduced to the average, so we can use them to recalculate the projection matrix.
        p_idx = 0
        state.ms_dict[bucket_index].clear()
        for tensor in compressed_tensors:
            n, m = tensor.shape
            _min = min(n, m)
            M = state.m_memory_dict[bucket_index][
                p_idx : p_idx + _min * _min
            ].view(_min, _min)
            state.ms_dict[bucket_index].append(M)
            p_idx += _min * _min
            if state.identity_proj:
                M.copy_(torch.eye(_min, dtype=dtype, device=device))
            else:
                U, _, Vh = torch.linalg.svd(tensor.float(), full_matrices=True)
                if n < m:
                    M.copy_(U.to(device).to(dtype))
                else:
                    M.copy_(Vh.to(device).to(dtype))
        
        if bucket.is_last():
            print(f"iter: {state.iter}, update projection matrix.")

        state.maybe_increase_iter(bucket)
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Step O: compute the difference between the input tensor and the local error tensor
    input_tensor.add_(state.error_dict[bucket_index], alpha=1.0)   # input_tensor = \nabla F_i + E_{i-1}
    input_tensor_cp = input_tensor.detach().clone()

    # Step 1: compute the projection of the input tensor
    def get_subspace_norm(ms, ct):
        # Unbatched, x is a 2D tensor
        total_scaler_size = sum([x.shape[1] for x in ms])
        scaler_memory = torch.empty(total_scaler_size, device=device, dtype=dtype)
        scalers = []
        scaler_index = 0

        if state.sigma_type == 0:
            for M, tensor in zip(ms, ct):
                n, m = tensor.shape
                check_generator_identity(state, state.generator[bucket_index])
                if n < m:
                    torch.norm(torch.mm(M.transpose(-2, -1), tensor), dim=1, out=scaler_memory[scaler_index : scaler_index + n])
                else:
                    torch.norm(torch.mm(tensor, M.transpose(-2, -1)), dim=0, out=scaler_memory[scaler_index : scaler_index + m])
                scalers.append(scaler_memory[scaler_index : scaler_index + n if n < m else scaler_index + m])
            # square itself before allreduce
            scaler_memory.mul_(scaler_memory)
            dist.all_reduce(scaler_memory, group=group_to_use, async_op=False)
            scaler_memory.div_(world_size)
        elif state.sigma_type == 1:
            for M, tensor in zip(ms, ct):
                n, m = tensor.shape
                if n < m:
                    rand_vector = torch.rand(m, 1, device = device, dtype=dtype, generator = state.generator[bucket_index])
                    torch.linalg.multi_dot([M.transpose(-2, -1), tensor, rand_vector], out=scaler_memory[scaler_index : scaler_index + n].view(-1, 1))
                else:
                    rand_vector = torch.rand(1, n, device = device, dtype=dtype, generator = state.generator[bucket_index])
                    torch.linalg.multi_dot([rand_vector, tensor, M.transpose(-2, -1)], out=scaler_memory[scaler_index : scaler_index + m].view(1, -1))
                scalers.append(scaler_memory[scaler_index : scaler_index + n if n < m else scaler_index + m])
            dist.all_reduce(scaler_memory, group=group_to_use, async_op=False)
            scaler_memory.div_(world_size)
            scaler_memory.abs_()
        elif state.sigma_type == 2:
            for M, tensor in zip(ms, ct):
                n, m = tensor.shape
                rand_matrix = torch.rand(n, m, device = device, dtype=dtype, generator = state.generator[bucket_index])
                if n < m:
                    # (M.transpose(-2, -1) @ tensor * rand_matrix).sum(dim=1, out=scaler_memory[scaler_index : scaler_index + n])
                    torch.sum(M.transpose(-2, -1) @ tensor * rand_matrix, dim=1, out=scaler_memory[scaler_index : scaler_index + n])
                else:
                    # (tensor @ M.transpose(-2, -1) * rand_matrix).sum(dim=0, out=scaler_memory[scaler_index : scaler_index + m])
                    torch.sum(tensor @ M.transpose(-2, -1) * rand_matrix, dim=0, out=scaler_memory[scaler_index : scaler_index + m])
                scalers.append(scaler_memory[scaler_index : scaler_index + n if n < m else scaler_index + m])
            dist.all_reduce(scaler_memory, group=group_to_use, async_op=False)
            scaler_memory.div_(world_size)
            scaler_memory.abs_()

        return scalers

    uncompressed_tensors, compressed_tensors, total_Ms_size, total_Rs_size = divide_tensors_to_compress_subspace_single(bucket, state)
    if not state.random:
        scalers = get_subspace_norm(state.ms_dict[bucket_index], compressed_tensors)
    else:
        scalers = [None for _ in range(len(compressed_tensors))]

    # Step 2: Simulate the compression and decompression process
    for M, tensor, scaler in zip(state.ms_dict[bucket_index], compressed_tensors, scalers):
        n, m = tensor.shape
        if not state.random:
            indices = torch.argsort(scaler, descending=True)[:state.matrix_approximation_rank]
        else:
            logger.info(f"Rank[{dist.get_rank()}] Using random projection!")
            indices = torch.randint(0, min(n, m), (state.matrix_approximation_rank,), device=device, generator=state.generator[bucket_index])
        if n < m:
            P = M[:, indices]
            tensor.copy_(torch.linalg.multi_dot([P, P.t(), tensor]))
        else:
            P = M[indices, :]
            tensor.copy_(torch.linalg.multi_dot([tensor, P.t(), P]))
            
    # Step 3: compute the difference between the compressed tensor and the input tensor
    state.error_dict[bucket_index].lerp_(input_tensor_cp - input_tensor, 1 - state.beta_ef)
    if dist.get_rank() == 0:
        print(f"Rank[{dist.get_rank()}] Iter[{state.iter}], bucket index[{bucket_index}], compress error: {torch.norm(input_tensor_cp - input_tensor).item()}")
    
    # Step 3.5: Scaling low rank part
    for tensor in compressed_tensors:
        tensor.mul_(state.scale)

    # Step 4: allreduce the compressed tensor
    state.maybe_increase_iter(bucket)
    input_tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(input_tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )
   