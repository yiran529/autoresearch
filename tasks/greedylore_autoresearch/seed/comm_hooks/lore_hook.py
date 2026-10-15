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


class LoreState(HookState):
    
    def __init__(self,
        process_group: dist.ProcessGroup,
        matrix_approximation_rank: int = 8,
        update_proj_gap: int = 200,
        start_compress_iter: int = 10,
        min_compression_rate: int = 2.0,
        use_error_feedback: str = "ef14",
        batch_tensors_with_same_shape: bool = True,
        uncompressed_names: List[str] = ["classifier", "embedding"],
        scale: int = 1.0,
        random_seed: int = 0,
    ):
        super().__init__(process_group)
        
        self.total_bit_before_compression = 0
        self.total_bit_after_compression = 0
        self.total_numel_before_compression = 0
        self.total_numel_after_compression = 0

        self.iter = 0
        self.start_compress_iter = start_compress_iter
        self.compressor_name = f"Lore"

        # EF-Lore parameters
        self.matrix_approximation_rank = matrix_approximation_rank
        self.update_proj_gap = update_proj_gap
        self.min_compression_rate = min_compression_rate
        self.batch_tensors_with_same_shape = batch_tensors_with_same_shape
        self.uncompressed_names = uncompressed_names
        self.scale = scale
    
        self.update_tracker_every_gap = False
        self.svd_on_cpu = False

        # Error feedback
        self.use_error_feedback = use_error_feedback
        self.checking = False

        # state dict
        self.error_dict: Dict[int, torch.Tensor] = {}       # store 1D error tensor for EF14 and EF21
        self.global_error_dict: Dict[int, torch.Tensor] = {}    # store 1D error tensor for EF21
        self.p_memory_dict: Dict[int, torch.Tensor] = {}
        self.ps_dict: Dict[int, List[torch.Tensor]] = {}

def check_index_identity(bucket):
    input_tensor = bucket.buffer()
    tensors = bucket.gradients()
    idx = 0
    for tensor in tensors:
        assert torch.allclose(input_tensor[idx : idx + tensor.numel()], tensor.view(-1)), f"Tensor {idx} is not the same, {input_tensor[idx : idx + tensor.numel()]} != {tensor.view(-1)}"
        idx += tensor.numel()

def check_shape_to_tensor_identity(shape_to_tensor1, shape_to_tensor2):
    assert len(shape_to_tensor1) == len(shape_to_tensor2), "The number of shapes should be equal."
    for shape, tensors1 in shape_to_tensor1.items():
        tensors2 = shape_to_tensor2[shape]
        assert len(tensors1) == len(tensors2), "The number of tensors should be equal."

def _should_compress(
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
    compressed_size = (num_rows + num_cols) * matrix_approximation_rank
    return (
        compressed_size * min_compression_rate < uncompressed_size,
        uncompressed_size,
        compressed_size,
    )

def divide_tensors_to_compress(bucket: dist.GradBucket, state: LoreState):
    uncompressed_tensors = []
    shape_to_tensors = defaultdict(list)
    total_Ps_size, total_Rs_size = 0, 0
    tensors, params = bucket.gradients(), bucket.parameters()
    for tensor, param in zip(tensors, params):
        # TODO: add branch when n is much larger than m
        matrix = tensor.view(tensor.shape[0], -1)
        n, m = matrix.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        should_compress, uncompressed_size, compressed_size = _should_compress(
            n, m, matrix_approximation_rank, state.min_compression_rate
        )
        # Don't compress if embedding in name or classifier in name
        if any(name in state.param_to_name[param] for name in state.uncompressed_names):
            if state.iter == state.start_compress_iter:
                logger.info(f"Skip compressing {state.param_to_name[param]}")
                print(f"Skip compressing {state.param_to_name[param]}")
            should_compress = False
        state.total_numel_before_compression += uncompressed_size
        state.total_bit_before_compression += uncompressed_size * dtype_bits(tensor)
        if should_compress:
            shape_to_tensors[matrix.shape].append(matrix)
            total_Ps_size += n * matrix_approximation_rank
            total_Rs_size += matrix_approximation_rank * m
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

def lore_hook_ef21(
    state: LoreState, bucket: dist.GradBucket
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
        # 1. Set up local error tensor for EF21
        # E_i = \nabla F_i
        if bucket_index in state.error_dict:
            if state.update_tracker_every_gap or state.iter == state.start_compress_iter:
                state.error_dict[bucket_index].copy_(input_tensor)        
        else:
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.clone(input_tensor).view(-1).detach()

        # 2. Allreduce the difference to calculate the average.
        dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
        input_tensor.div_(world_size)
        
        # 3. Set up global error tensor for EF21
        # \overline{E_i} = \overline{\nabla F_i}
        if bucket_index in state.global_error_dict:
            if state.update_tracker_every_gap or state.iter == state.start_compress_iter:
                state.global_error_dict[bucket_index].copy_(input_tensor)
        else:
            logger.info("A zero tensor of length %s that represents global error is created.", total_length)
            state.global_error_dict[bucket_index] = torch.clone(input_tensor).view(-1).detach()

        # 4. Divide all the tensors into two groups, reate dict related to projection matrix
        uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress(bucket, state)

        if bucket_index not in state.p_memory_dict:
            state.p_memory_dict[bucket_index] = torch.empty(
                total_Ps_size, device=device, dtype=dtype
            )
        if bucket_index not in state.ps_dict:
            state.ps_dict[bucket_index] = list()

        # 5. Tensors have been allreduced to the average, so we can use them to recalculate the projection matrix.
        p_idx = 0
        state.ps_dict[bucket_index].clear()
        batched_tensors_to_compress = list(maybe_batched_tensors_to_compress(stt=shape_to_tensors, state=state))
        for tensor in batched_tensors_to_compress:
            # 5.1 fetch Ps and store them in the state
            batch_size, n, m = tensor.shape
            matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
            batched_P = state.p_memory_dict[bucket_index][
                p_idx : p_idx + batch_size * n * matrix_approximation_rank
                ].view(batch_size, n, matrix_approximation_rank)
            state.ps_dict[bucket_index].append(batched_P)
            p_idx += batch_size * n * matrix_approximation_rank
            # 5.2 SVD tensors and update Ps
            # cast to float
            if state.svd_on_cpu:
                batched_U, _, batched_Vh = torch.linalg.svd(tensor.cpu().float(), full_matrices=False)
            else:
                batched_U, _, batched_Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
            batched_P.copy_(batched_U[:, :, :matrix_approximation_rank].to(device).to(dtype))

        if bucket.is_last():
            logger.info(f"iter: {state.iter}, update projection matrix.")


        state.maybe_increase_iter(bucket)
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Step O: compute the difference between the input tensor and the local error tensor
    input_tensor.add_(state.error_dict[bucket_index], alpha=-1.0)   # input_tensor = \nabla F_i - E_{i-1}

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    # 1D tensors will be eliminated from the compression process because min_compression_rate < 2.
    uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress(bucket, state)

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
    
    # Create Rs that point to the allocated memory.
    rs = []
    r_idx = 0
    accum_norm_error = 0
    r_memory = torch.empty(total_Rs_size, device=device, dtype=dtype)
    for batched_P, batched_tensor in zip(state.ps_dict[bucket_index], batched_tensors_to_compress):
        # 1. Fetch Rs and link them to the allocated memory
        batch_size, n, m = batched_tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        batched_R = r_memory[
            r_idx : r_idx + batch_size * matrix_approximation_rank * m
            ].view(batch_size, matrix_approximation_rank, m)
        r_idx += batch_size * matrix_approximation_rank * m
        
        # 2. Compute compressed tensor and store them in Rs
        torch.bmm(batched_P.transpose(1, 2), batched_tensor, out=batched_R)
        rs.append(batched_R)

        # 3. Update the compressed difference in input tensor for EF21 local error update
        # input_tensor = C[\nabla F_i - E_{i-1}]
        batched_D = torch.bmm(batched_P, batched_R)
        accum_norm_error += torch.norm(batched_tensor - batched_D).item() ** 2
        original_tensor_list = shape_to_tensors[batched_tensor.shape[1:]]
        for i, original_tensor in enumerate(original_tensor_list):
            original_tensor.copy_(batched_D[i])

    if bucket_index == 12 and dist.get_rank() == 0 and not update_projection:
        logger.info(f"iter: {state.iter}, ef21 accum local error: {accum_norm_error ** 0.5}")
        logger.info(f"compressed tensor num: {sum([len(x) for x in shape_to_tensors.values()])}")
        logger.info(f"ps's shapes: {[x.shape for x in state.ps_dict[bucket_index]]}")
        logger.info(f"xs's shapes: {[x.shape for x in batched_tensors_to_compress]}")
    # if bucket_index == 12:
    #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], local error: {state.error_dict[bucket_index][-5:]}")
    #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], global error: {state.global_error_dict[bucket_index][-5:]}")
    #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], compressed diff: {input_tensor[-5:]}")
    
    # 4. Update local error tensor
    # E_i = E_{i-1} + C[\nabla F_i - E_{i-1}]
    state.error_dict[bucket_index].add_(input_tensor, alpha=1.0)
    # if bucket_index == 12:
    #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], updated local error: {state.error_dict[bucket_index][-5:]}")

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
        for batched_P, batched_R, batched_tensor in zip(state.ps_dict[bucket_index], rs, batched_tensors_to_compress):
            torch.bmm(batched_P, batched_R, out=batched_tensor)
        del r_memory

        # Copy batched tensors back to original buffer.
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

        # Now, input_tensor = \overline{C[\nabla F_i - E_{i-1}]}
        state.global_error_dict[bucket_index].add_(input_tensor) # \overline{E}_i = \overline{E}_{i-1} \overline{C[\nabla F_i - E_{i-1}]}
        # if bucket_index == 12:
        #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], averaged compressed diff: {input_tensor[-5:]}")
        #     logger.info(f"Rank[{dist.get_rank()}] Iter[{state.iter}], updated global error: {state.global_error_dict[bucket_index][-5:]}")

        # if bucket_index == 12:
        #     check_error_identity(state.error_dict[bucket_index], state.global_error_dict[bucket_index], group=group_to_use, iteration=state.iter)
        state.maybe_increase_iter(bucket)

        return state.global_error_dict[bucket_index]

    return (
        allreduce_contiguous_uncompressed_tensors_fut.then(
            unpack_uncompressed_tensors_and_allreduce_rs
        ).then(decompress)
    )

def lore_hook_ef14(
    state: LoreState, bucket: dist.GradBucket
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
        # 1. Set up local error tensor for EF14
        # E_i = 0
        if bucket_index in state.error_dict:
            state.error_dict[bucket_index].zero_()
        else:
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)

        # 2. Allreduce the difference to calculate the average.
        dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
        input_tensor.div_(world_size)

        # 3. Divide all the tensors into two groups, reate dict related to projection matrix
        uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress(bucket, state)

        if bucket_index not in state.p_memory_dict:
            state.p_memory_dict[bucket_index] = torch.empty(
                total_Ps_size, device=device, dtype=dtype
            )
        if bucket_index not in state.ps_dict:
            state.ps_dict[bucket_index] = list()

        # 4. Tensors have been allreduced to the average, so we can use them to recalculate the projection matrix.
        p_idx = 0
        state.ps_dict[bucket_index].clear()
        batched_tensors_to_compress = list(maybe_batched_tensors_to_compress(stt=shape_to_tensors, state=state))
        for tensor in batched_tensors_to_compress:
            # 4.1 fetch Ps and store them in the state
            batch_size, n, m = tensor.shape
            matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
            batched_P = state.p_memory_dict[bucket_index][
                p_idx : p_idx + batch_size * n * matrix_approximation_rank
                ].view(batch_size, n, matrix_approximation_rank)
            state.ps_dict[bucket_index].append(batched_P)
            p_idx += batch_size * n * matrix_approximation_rank
            # 4.2 SVD tensors and update Ps
            # cast to float
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
            batched_P.copy_(batched_U[:, :, :matrix_approximation_rank].to(device).to(dtype))

        if bucket.is_last():
            logger.info(f"iter: {state.iter}, update projection matrix.")

        state.maybe_increase_iter(bucket)
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Step O: compute the difference between the input tensor and the local error tensor
    input_tensor.add_(state.error_dict[bucket_index], alpha=1.0)   # input_tensor = \nabla F_i + E_{i-1}
    state.error_dict[bucket_index].copy_(input_tensor) # error = \nabla F_i + E_{i-1} (store the uncompressed tensor temporarily)

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    # 1D tensors will be eliminated from the compression process because min_compression_rate < 2.
    uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress(bucket, state)

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
    
    # Create Rs that point to the allocated memory.
    rs = []
    r_idx = 0
    accum_norm_error = 0
    r_memory = torch.empty(total_Rs_size, device=device, dtype=dtype)
    for batched_P, batched_tensor in zip(state.ps_dict[bucket_index], batched_tensors_to_compress):
        # 1. Fetch Rs and link them to the allocated memory
        batch_size, n, m = batched_tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        batched_R = r_memory[
            r_idx : r_idx + batch_size * matrix_approximation_rank * m
            ].view(batch_size, matrix_approximation_rank, m)
        r_idx += batch_size * matrix_approximation_rank * m
        
        # 2. Compute compressed tensor and store them in Rs
        torch.bmm(batched_P.transpose(1, 2), batched_tensor, out=batched_R)
        rs.append(batched_R)

        # 3. Update the compressed difference in input tensor for EF14 local error update
        # input_tensor = C[\nabla F_i + E_{i-1}]
        batched_D = torch.bmm(batched_P, batched_R)
        original_tensor_list = shape_to_tensors[batched_tensor.shape[1:]]
        for i, original_tensor in enumerate(original_tensor_list):
            accum_norm_error += torch.norm(original_tensor - batched_D[i]).item() ** 2
            original_tensor.copy_(batched_D[i])

    if bucket_index == 12 and dist.get_rank() == 0 and not update_projection:
        logger.info(f"iter: {state.iter}, ef14 accum local error: {accum_norm_error ** 0.5}")
        # logger.info(f"compressed tensor num: {sum([len(x) for x in shape_to_tensors.values()])}")
        # logger.info(f"ps's shapes: {[x.shape for x in state.ps_dict[bucket_index]]}")
        # logger.info(f"xs's shapes: {[x.shape for x in batched_tensors_to_compress]}")
    
    # 4. Update local error tensor
    # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
    state.error_dict[bucket_index].add_(input_tensor, alpha=-1.0)


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
        r_memory = fut.value()
        for batched_P, batched_R, batched_tensor in zip(state.ps_dict[bucket_index], rs, batched_tensors_to_compress):
            torch.bmm(batched_P, batched_R, out=batched_tensor)
        del r_memory

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

def lore_hook_noef(
    state: LoreState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:

    # check if the key is ready in precond adam and maybe accumulate momentum on bucket
    state.maybe_accumulate_momentum_on_bucket(bucket)

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor; Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()

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
        # 1. Allreduce the difference to calculate the average.
        dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
        input_tensor.div_(world_size)

        # 2. Divide all the tensors into two groups, reate dict related to projection matrix
        uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress(bucket, state)

        if bucket_index not in state.p_memory_dict:
            state.p_memory_dict[bucket_index] = torch.empty(
                total_Ps_size, device=device, dtype=dtype
            )
        if bucket_index not in state.ps_dict:
            state.ps_dict[bucket_index] = list()

        # 3. Tensors have been allreduced to the average, so we can use them to recalculate the projection matrix.
        p_idx = 0
        state.ps_dict[bucket_index].clear()
        batched_tensors_to_compress = list(maybe_batched_tensors_to_compress(stt=shape_to_tensors, state=state))
        for tensor in batched_tensors_to_compress:
            # 3.1 fetch Ps and store them in the state
            batch_size, n, m = tensor.shape
            matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
            batched_P = state.p_memory_dict[bucket_index][
                p_idx : p_idx + batch_size * n * matrix_approximation_rank
                ].view(batch_size, n, matrix_approximation_rank)
            state.ps_dict[bucket_index].append(batched_P)
            p_idx += batch_size * n * matrix_approximation_rank
            # 3.2 SVD tensors and update Ps
            # cast to float
            if state.svd_on_cpu:
                batched_U, _, batched_Vh = torch.linalg.svd(tensor.cpu().float(), full_matrices=False)
            else:
                batched_U, _, batched_Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
            batched_P.copy_(batched_U[:, :, :matrix_approximation_rank].to(device).to(dtype))

        if bucket.is_last():
            logger.info(f"iter: {state.iter}, update projection matrix.")

        state.maybe_increase_iter(bucket)
        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(input_tensor)
        return fut

    # Step I: Divide all the tensors into two groups,
    # one will be compressed before allreduce and the other will be directly allreduced without compression.
    # 1D tensors will be eliminated from the compression process because min_compression_rate < 2.
    uncompressed_tensors, shape_to_tensors, total_Ps_size, total_Rs_size = divide_tensors_to_compress(bucket, state)

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
    
    # Create Rs that point to the allocated memory.
    rs = []
    r_idx = 0
    accum_norm_error = 0
    r_memory = torch.empty(total_Rs_size, device=device, dtype=dtype)
    for batched_P, batched_tensor in zip(state.ps_dict[bucket_index], batched_tensors_to_compress):
        # 1. Fetch Rs and link them to the allocated memory
        batch_size, n, m = batched_tensor.shape
        matrix_approximation_rank = min(n, m, state.matrix_approximation_rank)
        batched_R = r_memory[
            r_idx : r_idx + batch_size * matrix_approximation_rank * m
            ].view(batch_size, matrix_approximation_rank, m)
        r_idx += batch_size * matrix_approximation_rank * m
        
        # 2. Compute compressed tensor and store them in Rs
        torch.bmm(batched_P.transpose(1, 2), batched_tensor, out=batched_R)
        rs.append(batched_R)

        # input_tensor = C[\nabla F_i]
        batched_D = torch.bmm(batched_P, batched_R)
        original_tensor_list = shape_to_tensors[batched_tensor.shape[1:]]
        for i, original_tensor in enumerate(original_tensor_list):
            accum_norm_error += torch.norm(original_tensor - batched_D[i]).item() ** 2
            original_tensor.copy_(batched_D[i])

    
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
        r_memory = fut.value()
        for batched_P, batched_R, batched_tensor in zip(state.ps_dict[bucket_index], rs, batched_tensors_to_compress):
            torch.bmm(batched_P, batched_R, out=batched_tensor)
        del r_memory

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


def lore_hook(
    state: LoreState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    
    if state.use_error_feedback == "ef21":
        return lore_hook_ef21(state, bucket)
    elif state.use_error_feedback == "ef14":
        return lore_hook_ef14(state, bucket)
    elif state.use_error_feedback == "noef":
        return lore_hook_noef(state, bucket)
    else:
        raise ValueError("The error feedback method should be one of 'ef21', 'ef14', 'noef'.")

if __name__ == "__main__":
    pass