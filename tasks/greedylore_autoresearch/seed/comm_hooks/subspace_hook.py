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


class SubspaceState(HookState):
    
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
        scale: int = 1.0,
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

def check_index_identity(bucket):
    input_tensor = bucket.buffer()
    tensors = bucket.gradients()
    idx = 0
    for tensor in tensors:
        assert torch.allclose(input_tensor[idx : idx + tensor.numel()], tensor.view(-1)), f"Tensor {idx} is not the same, {input_tensor[idx : idx + tensor.numel()]} != {tensor.view(-1)}"
        idx += tensor.numel()

def check_generator_identity(state: SubspaceState, generator: torch.Generator):
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

def divide_tensors_to_compress_subspace(bucket: dist.GradBucket, state: SubspaceState):
    uncompressed_tensors = []
    shape_to_tensors = defaultdict(list)
    total_Ps_size, total_Rs_size = 0, 0
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
            shape_to_tensors[matrix.shape].append(matrix)
            total_Ps_size += _min * _min
            total_Rs_size += _max * matrix_approximation_rank
            state.total_numel_after_compression += compressed_size
            state.total_bit_after_compression += compressed_size * dtype_bits(tensor)
        else:
            uncompressed_tensors.append(tensor)
            state.total_numel_after_compression += uncompressed_size
            state.total_bit_after_compression += uncompressed_size * dtype_bits(tensor)
    return uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size

def maybe_batched_tensors_to_compress(stt, state):
    for tensors in stt.values():
        if state.batch_tensors_with_same_shape:
            batch_size = len(tensors)
            if batch_size == 1:
                # Use the original tensor to avoid copy.
                yield tensors[0].unsqueeze(0)
            else:
                yield torch.stack(tensors)
        else:
            for tensor in tensors:
                yield tensor.unsqueeze(0) 


def estimate_subspace_scores(bases, gradients, generator, process_group):
    """Estimate each shared basis direction's global gradient energy.

    Each direction uses an independent Gaussian probe. The signed local
    estimates are averaged before squaring, as in GreedyLore Algorithm 2.
    """
    local_scores = []
    for basis, gradient in zip(bases, gradients):
        rows, columns = gradient.shape[1:]
        if rows < columns:
            projected = torch.bmm(basis.transpose(1, 2).float(), gradient.float())
            probes = torch.randn(projected.shape, device=gradient.device,
                                 generator=generator)
            local_scores.append((projected * probes).sum(dim=2))
        else:
            projected = torch.bmm(gradient.float(), basis.transpose(1, 2).float())
            probes = torch.randn(projected.shape, device=gradient.device,
                                 generator=generator)
            local_scores.append((projected * probes).sum(dim=1))

    if not local_scores:
        return []
    sizes = [score.numel() for score in local_scores]
    scores = torch.cat([score.reshape(-1) for score in local_scores])
    if process_group is not None and dist.get_world_size(process_group) > 1:
        dist.all_reduce(scores, group=process_group)
        scores.div_(dist.get_world_size(process_group))
    return [score.square().view_as(original) for score, original in
            zip(scores.split(sizes), local_scores)]

def subspace_hook_ef14(
    state: SubspaceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # check if the key is ready in precond adam
    state.maybe_accumulate_momentum_on_bucket(bucket)

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
                # logger.info("Inherit local error tensor from the previous iteration.")
                # print(f"Rank[{dist.get_rank()}] Iter[{state.iter}], bi:{bucket_index}, error: {torch.norm(state.error_dict[bucket_index])}")
                pass
            else:
                state.error_dict[bucket_index].zero_()
        else:
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)

        # 2. Allreduce the difference to calculate the average.
        dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
        input_tensor.div_(world_size)

        # 3. Divide all the tensors into two groups, reate dict related to projection matrix
        uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress_subspace(bucket, state)

        if bucket_index not in state.m_memory_dict:
            state.m_memory_dict[bucket_index] = torch.empty(
                total_Ps_size, device=device, dtype=dtype
            )
        if bucket_index not in state.ms_dict:
            state.ms_dict[bucket_index] = list()

        # 4. Tensors have been allreduced to the average, so we can use them to recalculate the projection matrix.
        p_idx = 0
        state.ms_dict[bucket_index].clear()
        batched_tensors_to_compress = list(maybe_batched_tensors_to_compress(stt=shape_to_tensors, state=state))
        for tensor in batched_tensors_to_compress:
            # 4.1 fetch Ps and store them in the state
            batch_size, n, m = tensor.shape
            _min = min(n, m)
            batched_M = state.m_memory_dict[bucket_index][
                p_idx : p_idx + batch_size * _min * _min
                ].view(batch_size, _min, _min)
            state.ms_dict[bucket_index].append(batched_M)
            p_idx += batch_size * _min * _min
            # 4.2 SVD tensors and update Ps
            # cast to float
            if state.identity_proj:
                batched_U = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, n, n)
                batched_Vh = torch.eye(m, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, m, m)
            else:
                try:
                    if state.svd_on_cpu:
                        batched_U, _, batched_Vh = torch.linalg.svd(tensor.cpu().float(), full_matrices=False)
                    else:
                        batched_U, _, batched_Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
                except Exception as e:
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], tensor shape: {tensor.shape}")
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], tensor: {tensor}")
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], batched_U: {batched_U}")
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], batched_Vh: {batched_Vh}")
                    torch.save(tensor.float(), f"tensor_{dist.get_rank()}_{state.iter}.pt")
                    raise e
            if n < m:
                batched_M.copy_(batched_U.to(device).to(dtype))
            else:
                batched_M.copy_(batched_Vh.to(device).to(dtype))

        if bucket.is_last():
            logger.info(f"iter: {state.iter}, update projection matrix.")

        state.maybe_increase_iter(bucket)
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Step O: compute the difference between the input tensor and the local error tensor
    input_tensor.add_(state.error_dict[bucket_index], alpha=1.0)   # input_tensor = \nabla F_i + E_{i-1}
    input_tensor_cp = input_tensor.detach().clone()
    # state.error_dict[bucket_index].copy_(input_tensor) # error = \nabla F_i + E_{i-1} (store the uncompressed tensor temporarily)

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    # 1D tensors will be eliminated from the compression process because min_compression_rate < 2.
    uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress_subspace(bucket, state)

    # Step II: Handle uncompressed tensors.
    # Allocate contiguous memory for these tensors to allreduce efficiently.
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step III: Handle the tensors that should be compressed.
    # This function decides whether to batch tensors with same shape or not according to the argument,
    # so the following process could share the same code.
    batched_tensors_to_compress = list(maybe_batched_tensors_to_compress(stt=shape_to_tensors, state=state))
    
    if not state.random:
        scalers = estimate_subspace_scores(
            state.ms_dict[bucket_index], batched_tensors_to_compress,
            state.generator[bucket_index], group_to_use,
        )
        
    # Create Rs that point to the allocated memory.
    rs, ps = [], []
    r_idx = 0
    accum_norm_error = 0
    r_memory = torch.empty(total_Rs_size, device=device, dtype=dtype)
    for i, (batched_M, batched_tensor) in enumerate(zip(state.ms_dict[bucket_index], batched_tensors_to_compress)):
        # 1. Fetch Rs and link them to the allocated memory
        batch_size, n, m = batched_tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        if n < m:
            batched_R = r_memory[
                r_idx : r_idx + batch_size * matrix_approximation_rank * m
                ].view(batch_size, matrix_approximation_rank, m)
            r_idx += batch_size * matrix_approximation_rank * m
            # 2. Select proj matrix, compute compressed tensor and store them in Rs
            assert batched_M.shape[1] == n and batched_M.shape[2] == n, f"batched_M shape: {batched_M.shape}, n: {n}"
            if state.random:
                indices = torch.randperm(n, generator=state.rng)[:matrix_approximation_rank]
                batched_P = batched_M[:, :, indices]
            else:
                # batched_M: [batch_size, n, n]
                scaler = scalers[i].view(batch_size, n) # [batch_size, n]
                indices = torch.argsort(scaler, dim=1, descending=True)[:, :matrix_approximation_rank] # [batch_size, matrix_approximation_rank]
                batched_P = torch.gather(batched_M, dim=2, index=indices.unsqueeze(1).expand(-1, n, matrix_approximation_rank))
            torch.bmm(batched_P.transpose(1, 2), batched_tensor, out=batched_R)            
        else:
            batched_R = r_memory[
                r_idx : r_idx + batch_size * n * matrix_approximation_rank
                ].view(batch_size, n, matrix_approximation_rank)
            r_idx += batch_size * n * matrix_approximation_rank
            # 2. Select proj matrix, compute compressed tensor and store them in Rs
            assert batched_M.shape[1] == m and batched_M.shape[2] == m, f"batched_M shape: {batched_M.shape}, m: {m}"
            if state.random:
                indices = torch.randperm(m, generator=state.rng)[:matrix_approximation_rank]
                batched_P = batched_M[:, indices, :]
            else:
                # batched_M: [batch_size, m, m]
                scaler = scalers[i].view(batch_size, m) # [batch_size, m]
                indices = torch.argsort(scaler, dim=1, descending=True)[:, :matrix_approximation_rank] # [batch_size, matrix_approximation_rank]
                batched_P = torch.gather(batched_M, dim=1, index=indices.unsqueeze(2).expand(-1, matrix_approximation_rank, m))
            torch.bmm(batched_tensor, batched_P.transpose(1, 2), out=batched_R)
        
        rs.append(batched_R)
        ps.append(batched_P)

        # 3. Update the compressed difference in input tensor for EF14 local error update
        # input_tensor = C[\nabla F_i + E_{i-1}]
        if n < m:
            batched_D = torch.bmm(batched_P, batched_R)
        else:
            batched_D = torch.bmm(batched_R, batched_P)
        original_tensor_list = shape_to_tensors[batched_tensor.shape[1:]]
        for i, original_tensor in enumerate(original_tensor_list):
            accum_norm_error += torch.norm(original_tensor - batched_D[i]).item() ** 2
            original_tensor.copy_(batched_D[i])

    # 4. Update local error tensor
    # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
    # state.error_dict[bucket_index].add_(input_tensor, alpha=-1.0)
    state.error_dict[bucket_index].lerp_(input_tensor_cp - input_tensor, 1 - state.beta_ef)
    # print(f"Iters: {state.iter}, Bucket: {bucket_index}, Accumulated norm error: {accum_norm_error}, Error: {torch.norm(input_tensor_cp - input_tensor)}")

    # Step IV: Start to allreduce the uncompressed tensors and compressed tensors.

    # This allreduce is only applied to uncompressed tensors,
    # so it should have been kicked off before the above computation on the compressed tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
        uncompressed_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_uncompressed_tensors_and_allreduce_rs(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor)
            )
            idx += tensor.numel()

        # Since these Ps will be orthogonalized later, no need to divide them by world size.
        return (
            dist.all_reduce(
                r_memory, group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        )

    def decompress(fut):
        # Decompress the compressed tensors.
        r_memory = fut.value().div_(world_size)
        for batched_P, batched_R, batched_tensor in zip(ps, rs, batched_tensors_to_compress):
            n, m = batched_tensor.shape[1], batched_tensor.shape[2]
            if n < m:
                torch.bmm(batched_P, batched_R, out=batched_tensor)
            else:
                torch.bmm(batched_R, batched_P, out=batched_tensor)
        del r_memory
        # TODO: check subspace compressor identity

        # Copy batched tensors back to original buffer.
        # input_tensor = \overline{C[\nabla F_i + E_{i-1}]}
        if state.batch_tensors_with_same_shape:
            for tensor in batched_tensors_to_compress:
                if tensor.shape[0] == 1:
                    # Skip tensor with batch_size == 1 since itself is the original tensor.
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])
                    original_tensor.mul_(state.scale)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        state.maybe_increase_iter(bucket)

        return input_tensor

    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_uncompressed_tensors_and_allreduce_rs
        ).then(decompress)
    )

def subspace_hook_noef(
    state: SubspaceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # check if the key is ready in precond adam
    state.maybe_accumulate_momentum_on_bucket(bucket)

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

    # Args for update
    update_projection = ((state.iter - state.start_compress_iter) % state.update_proj_gap == 0)
    
    # During the update_projection iteration, update the projection matrix.
    if update_projection:

        # 2. Allreduce the difference to calculate the average.
        dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
        input_tensor.div_(world_size)

        # 3. Divide all the tensors into two groups, reate dict related to projection matrix
        uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress_subspace(bucket, state)

        if bucket_index not in state.m_memory_dict:
            state.m_memory_dict[bucket_index] = torch.empty(
                total_Ps_size, device=device, dtype=dtype
            )
        if bucket_index not in state.ms_dict:
            state.ms_dict[bucket_index] = list()

        # 4. Tensors have been allreduced to the average, so we can use them to recalculate the projection matrix.
        p_idx = 0
        state.ms_dict[bucket_index].clear()
        batched_tensors_to_compress = list(maybe_batched_tensors_to_compress(stt=shape_to_tensors, state=state))
        for tensor in batched_tensors_to_compress:
            # 4.1 fetch Ps and store them in the state
            batch_size, n, m = tensor.shape
            _min = min(n, m)
            batched_M = state.m_memory_dict[bucket_index][
                p_idx : p_idx + batch_size * _min * _min
                ].view(batch_size, _min, _min)
            state.ms_dict[bucket_index].append(batched_M)
            p_idx += batch_size * _min * _min
            # 4.2 SVD tensors and update Ps
            # cast to float
            if state.identity_proj:
                batched_U = torch.eye(n, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, n, n)
                batched_Vh = torch.eye(m, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, m, m)
            else:            
                try:
                    if state.svd_on_cpu:
                        batched_U, _, batched_Vh = torch.linalg.svd(tensor.cpu().float(), full_matrices=False)
                    else:
                        batched_U, _, batched_Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
                except Exception as e:
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], tensor shape: {tensor.shape}")
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], tensor: {tensor}")
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], batched_U: {batched_U}")
                    logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], batched_Vh: {batched_Vh}")
                    torch.save(tensor.float(), f"tensor_{dist.get_rank()}_{state.iter}.pt")
                    raise e
            if n < m:
                batched_M.copy_(batched_U.to(device).to(dtype))
            else:
                batched_M.copy_(batched_Vh.to(device).to(dtype))

        if bucket.is_last():
            logger.info(f"iter: {state.iter}, update projection matrix.")

        state.maybe_increase_iter(bucket)
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    # 1D tensors will be eliminated from the compression process because min_compression_rate < 2.
    uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress_subspace(bucket, state)

    # Step II: Handle uncompressed tensors.
    # Allocate contiguous memory for these tensors to allreduce efficiently.
    uncompressed_tensors_memory = (
        torch.cat([tensor.view(-1) for tensor in uncompressed_tensors])
        if uncompressed_tensors
        else torch.tensor([], device=device, dtype=dtype)
    )

    # Step III: Handle the tensors that should be compressed.
    # This function decides whether to batch tensors with same shape or not according to the argument,
    # so the following process could share the same code.
    batched_tensors_to_compress = list(maybe_batched_tensors_to_compress(stt=shape_to_tensors, state=state))
    
    if not state.random:
        scalers = estimate_subspace_scores(
            state.ms_dict[bucket_index], batched_tensors_to_compress,
            state.generator[bucket_index], group_to_use,
        )
        
    # Create Rs that point to the allocated memory.
    rs, ps = [], []
    r_idx = 0
    accum_norm_error = 0
    r_memory = torch.empty(total_Rs_size, device=device, dtype=dtype)
    for i, (batched_M, batched_tensor) in enumerate(zip(state.ms_dict[bucket_index], batched_tensors_to_compress)):
        # 1. Fetch Rs and link them to the allocated memory
        batch_size, n, m = batched_tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        if n < m:
            batched_R = r_memory[
                r_idx : r_idx + batch_size * matrix_approximation_rank * m
                ].view(batch_size, matrix_approximation_rank, m)
            r_idx += batch_size * matrix_approximation_rank * m
            # 2. Select proj matrix, compute compressed tensor and store them in Rs
            assert batched_M.shape[1] == n and batched_M.shape[2] == n, f"batched_M shape: {batched_M.shape}, n: {n}"
            if state.random:
                indices = torch.randperm(n, generator=state.rng)[:matrix_approximation_rank]
                batched_P = batched_M[:, :, indices]
            else:
                # batched_M: [batch_size, n, n]
                scaler = scalers[i].view(batch_size, n) # [batch_size, n]
                indices = torch.argsort(scaler, dim=1, descending=True)[:, :matrix_approximation_rank] # [batch_size, matrix_approximation_rank]
                batched_P = torch.gather(batched_M, dim=2, index=indices.unsqueeze(1).expand(-1, n, matrix_approximation_rank))
            torch.bmm(batched_P.transpose(1, 2), batched_tensor, out=batched_R)            
        else:
            batched_R = r_memory[
                r_idx : r_idx + batch_size * n * matrix_approximation_rank
                ].view(batch_size, n, matrix_approximation_rank)
            r_idx += batch_size * n * matrix_approximation_rank
            # 2. Select proj matrix, compute compressed tensor and store them in Rs
            assert batched_M.shape[1] == m and batched_M.shape[2] == m, f"batched_M shape: {batched_M.shape}, m: {m}"
            if state.random:
                indices = torch.randperm(m, generator=state.rng)[:matrix_approximation_rank]
                batched_P = batched_M[:, indices, :]
            else:
                # batched_M: [batch_size, m, m]
                scaler = scalers[i].view(batch_size, m) # [batch_size, m]
                indices = torch.argsort(scaler, dim=1, descending=True)[:, :matrix_approximation_rank] # [batch_size, matrix_approximation_rank]
                batched_P = torch.gather(batched_M, dim=1, index=indices.unsqueeze(2).expand(-1, matrix_approximation_rank, m))
            torch.bmm(batched_tensor, batched_P.transpose(1, 2), out=batched_R)
        
        rs.append(batched_R)
        ps.append(batched_P)

        # 3. Update the compressed difference in input tensor for EF14 local error update
        # input_tensor = C[\nabla F_i + E_{i-1}]
        if n < m:
            batched_D = torch.bmm(batched_P, batched_R)
        else:
            batched_D = torch.bmm(batched_R, batched_P)
        original_tensor_list = shape_to_tensors[batched_tensor.shape[1:]]
        for i, original_tensor in enumerate(original_tensor_list):
            accum_norm_error += torch.norm(original_tensor - batched_D[i]).item() ** 2
            original_tensor.copy_(batched_D[i])

    if bucket_index == 12 and dist.get_rank() == 0 and not update_projection:
        logger.info(f"iter: {state.iter}, ef14 accum local error: {accum_norm_error ** 0.5}")
        # logger.info(f"compressed tensor num: {sum([len(x) for x in shape_to_tensors.values()])}")
        # logger.info(f"ps's shapes: {[x.shape for x in state.ms_dict[bucket_index]]}")
        # logger.info(f"xs's shapes: {[x.shape for x in batched_tensors_to_compress]}")
    

    # Step IV: Start to allreduce the uncompressed tensors and compressed tensors.

    # This allreduce is only applied to uncompressed tensors,
    # so it should have been kicked off before the above computation on the compressed tensors to hide more communication costs.
    # However, this somehow requires a separate future chain at this time.
    allreduce_contiguous_uncompressed_tensors_fut = dist.all_reduce(
        uncompressed_tensors_memory, group=group_to_use, async_op=True
    ).get_future()

    def unpack_uncompressed_tensors_and_allreduce_rs(fut):
        uncompressed_tensors_memory = fut.value()[0].div_(world_size)
        idx = 0
        for tensor in uncompressed_tensors:
            tensor.copy_(
                uncompressed_tensors_memory[idx : idx + tensor.numel()].view_as(tensor)
            )
            idx += tensor.numel()

        # Since these Ps will be orthogonalized later, no need to divide them by world size.
        return (
            dist.all_reduce(
                r_memory, group=group_to_use, async_op=True
            )
            .get_future()
            .wait()[0]
        )

    def decompress(fut):
        # Decompress the compressed tensors.
        r_memory = fut.value().div_(world_size)
        for batched_P, batched_R, batched_tensor in zip(ps, rs, batched_tensors_to_compress):
            n, m = batched_tensor.shape[1], batched_tensor.shape[2]
            if n < m:
                torch.bmm(batched_P, batched_R, out=batched_tensor)
            else:
                torch.bmm(batched_R, batched_P, out=batched_tensor)
        del r_memory
        # TODO: check subspace compressor identity

        # Copy batched tensors back to original buffer.
        # input_tensor = \overline{C[\nabla F_i + E_{i-1}]}
        if state.batch_tensors_with_same_shape:
            for tensor in batched_tensors_to_compress:
                if tensor.shape[0] == 1:
                    # Skip tensor with batch_size == 1 since itself is the original tensor.
                    continue
                original_tensors = shape_to_tensors[tensor.shape[1:]]
                for i, original_tensor in enumerate(original_tensors):
                    original_tensor.copy_(tensor[i])
                    original_tensor.mul_(state.scale)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        state.maybe_increase_iter(bucket)

        return input_tensor

    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_uncompressed_tensors_and_allreduce_rs
        ).then(decompress)
    )

def subspace_hook(
    state: SubspaceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    
    if state.use_error_feedback == "ef14":
        return subspace_hook_ef14(state, bucket)
    elif state.use_error_feedback == "noef":
        return subspace_hook_noef(state, bucket)
    else:
        raise ValueError("The error feedback method should be one of 'ef21', 'ef14', 'noef'.")

if __name__ == "__main__":
    pass
