# mypy: allow-untyped-defs
import torch
import torch.distributed as dist
import cupy
import numpy as np
import torch.nn.functional as F
import logging
from typing import Dict

import comm_hooks.default_hooks as default_hooks
from comm_hooks.utils import dtype_bits, tensor_bits, HookState, _get_allgather_out_list
from torch.utils.dlpack import to_dlpack, from_dlpack

logger = logging.getLogger(__name__)

def cupy_to_tensor(x):
    return from_dlpack(x.toDlpack())

def tensor_to_cupy(x):
    return cupy.fromDlpack(to_dlpack(x))

def pack_low_bit_tensor(x, bits):
    assert x.dtype == torch.uint8
    y = cupy.packbits(
        cupy.unpackbits(tensor_to_cupy(x)).reshape(*x.shape, 8)[..., -bits:]
    )
    y = cupy_to_tensor(y)
    return y

def unpack_low_bit_tensor(x, bits):
    y = cupy.packbits(cupy.pad(
        cupy.unpackbits(
            tensor_to_cupy(x)
        ).reshape(-1, bits),
        ((0,0), (8-bits, 0))
    ))
    y = cupy_to_tensor(y)
    return y

def _rounding(x, stochastic=False, minimum_stochastic_distance=0.2):
    if stochastic:
        x_floor = x.floor()
        th = x - x_floor
        if minimum_stochastic_distance > 0:
            th[th<minimum_stochastic_distance] = 0.
            th[th>1-minimum_stochastic_distance] = 1.
        pr = torch.rand_like(x)
        x_floor += (pr < th)
        return x_floor
    else:
        return x.round()

def _decompress_nbits(x, scale, bits):
    
    fbits = bits - 1
    
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1
    
    x = x.float() + clip_min
    # x = x + clip_min
    
    x = x / (clip_max+1) * scale
    
    return x

def _compress_nbits_by_bucket(x, bits, scale_method='max', bucket_size=512):
    
    fbits = bits - 1
    
    # x = x.view(-1, bucket_size).T
    x = x.view(bucket_size, -1)
    
    if scale_method == 'max':
        # issue: sensitive to outlier points
        scale = x.abs().amax([0], keepdims=True)
    elif scale_method == 'l2':
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean([0], keepdims=True).sqrt() * 2 
    else:
        raise Exception('unkonwn scale method.')
    # fp16 should be enough
    # scale = scale.half()
    x = x / (scale + 1e-6)
    
    x = x.ldexp(torch.tensor(fbits))
    clip_min = -(1<<fbits)
    clip_max = (1<<fbits)-1

    x = _rounding(x)
    x = x.clip(clip_min, clip_max)
    
    x = x - clip_min
    x = x.type(torch.uint8)
    
    return x, scale

def _decompress_signsgd(x, scale, bits=1):
    x = x.float()
    x.mul_(2).add_(-1).mul_(scale)
    return x

def _compress_signsgd_by_bucket(x:torch.Tensor, bits=1, scale_method='max', bucket_size=512):
    
    # x = x.view(-1, bucket_size).T
    x = x.view(bucket_size, -1)
    
    if scale_method == 'max':
        # issue: sensitive to outlier points
        scale = x.abs().amax([0], keepdims=True)
    elif scale_method == 'l2':
        # ~95% confidence interval for normal distribution
        scale = x.pow(2).mean([0], keepdims=True).sqrt() * 2 
    else:
        raise Exception('unkonwn scale method.')
    
    x.sign_().clip_(0, 1)

    x = x.type(torch.uint8)

    return x, scale

def compress_flexible_nbits_by_bucket(x, bits, scale_method='max', bucket_size=512):
    # support any bits
    # CUDA only
    
    # padding
    pad_size = bucket_size - x.numel() % bucket_size
    if pad_size < bucket_size:
        x = F.pad(x.view(-1), (0, pad_size), "constant", 0)
    
    if bits > 1:
        x, scale = _compress_nbits_by_bucket(x, bits=bits, scale_method=scale_method, bucket_size=bucket_size)
    else:
        x, scale = _compress_signsgd_by_bucket(x, bits=bits, scale_method=scale_method, bucket_size=bucket_size)
    
    x = pack_low_bit_tensor(x, bits)
    
    return x, scale

def decompress_flexible_nbits_by_bucket(x, scale, bits, original_shape, bucket_size=512):
    # support any bits, but need to know original_shape
    # CUDA only
    x = unpack_low_bit_tensor(x, bits)
    
    # x = x.view(-1, bucket_size).T
    x = x.view(bucket_size, -1)

    if bits > 1:
        x = _decompress_nbits(x, scale, bits=bits)  # scale shape: bucket_num x 1
    else:
        x = _decompress_signsgd(x, scale, bits=bits)
    

    # unpadding
    x = x.view(-1)[:original_shape.numel()].view(original_shape)
    
    return x

class FlexQuantState(HookState):
    
    def __init__(self, process_group: dist.ProcessGroup, **kwargs):
        super().__init__(process_group)
        self.total_bit_before_compression = 0
        self.total_bit_after_compression = 0
        self.quantization_bits = kwargs.get('quantization_bits', 8)
        self.bucket_size = kwargs.get('bucket_size', 512)
        self.scale_method = kwargs.get('scale_method', 'max')

        self.iter = 0
        self.start_compress_iter = 10
        self.util_dict = {}
        self.compressor_name = f"{self.quantization_bits} bit quantization"
        
        # error feedback
        self.use_error_feedback = kwargs.get("use_error_feedback", "noef")
        self.global_error_dict: Dict[int, torch.Tensor] = {}
        self.error_dict: Dict[int, torch.Tensor] = {}

def flex_quant_hook_sync(
    state: FlexQuantState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer

    # check if the key is ready in precond adam
    state.maybe_accumulate_momentum_on_bucket(bucket)

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    rank = dist.get_rank(group=group_to_use)

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor)

    # Apply sparse after `start_compress_iter` iterations.
    device = input_tensor.device
    dtype = input_tensor.dtype

    # Get dtype and device.
    original_shape = input_tensor.shape

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index()
    total_length = input_tensor.shape[0]
    if state.use_error_feedback =="ef14":
        if bucket_index in state.error_dict:
            # input_tensor = \nabla F_i + E_{i-1}
            input_tensor.add_(state.error_dict[bucket_index])
        else:
            # E_{i-1} is not available, create a zero tensor.
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
    elif state.use_error_feedback == "ef21":
        if bucket_index in state.error_dict:
            # input_tensor = \nabla F_i - E_{i-1}
            input_tensor.add_(state.error_dict[bucket_index], alpha=-1.0)
        else:
            # E_0 = \nabla F_0 
            logger.info("A tensor of length %s that represents local/global error is created.", total_length)
            state.error_dict[bucket_index] = torch.clone(input_tensor).detach()
            # allreduce \nabla F_0
            dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
            input_tensor.div_(world_size)
            # \overline{E_0} = \overline{\nabla F_0}
            state.global_error_dict[bucket_index] = torch.clone(input_tensor).detach()
            # reset the full input tensor
            state.maybe_increase_iter(bucket)
            fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
            fut.set_result(input_tensor)
            return fut

    # Compress the tensor and get the scale.
    quantized_tensor, scale = compress_flexible_nbits_by_bucket(input_tensor.clone().detach(), bits=state.quantization_bits, scale_method=state.scale_method, bucket_size=state.bucket_size)

    if state.use_error_feedback in ["ef14", "ef21"]:
        # EF14: dequantized_tensor = C[\nabla F_i + E_{i-1}]
        # EF21: dequantized_tensor = C[\nabla F_i - E_{i-1}]
        dequantized_tensor = decompress_flexible_nbits_by_bucket(quantized_tensor, scale, bits=state.quantization_bits, original_shape=original_shape, bucket_size=state.bucket_size)
        if state.use_error_feedback == "ef14":
            state.error_dict[bucket_index] = input_tensor - dequantized_tensor  # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
        elif state.use_error_feedback == "ef21":
            state.error_dict[bucket_index].add_(dequantized_tensor, alpha=1.0)  # E_i = E_{i-1} + C[\nabla F_i - E_{i-1}]
            

    # Allocate memory and all gather the scales and zeros.
    all_ranks_scale = _get_allgather_out_list(scale, world_size)
    all_ranks_quantized_tensor = _get_allgather_out_list(quantized_tensor, world_size)

    dist.all_gather(all_ranks_scale, scale, group=group_to_use, async_op=False)
    dist.all_gather(
        all_ranks_quantized_tensor,
        quantized_tensor,
        group=group_to_use,
        async_op=False,
    )

    # Zero the input tensor.
    input_tensor.zero_()
    # Using previously allgathered scales, dequantize gradient tensors locally and then aggregate them.
    for scale, quantized_tensor in zip(all_ranks_scale, all_ranks_quantized_tensor):
        dequantized_tensor = decompress_flexible_nbits_by_bucket(quantized_tensor, scale, bits=state.quantization_bits, original_shape=original_shape, bucket_size=state.bucket_size)
        input_tensor.add_(dequantized_tensor)
    input_tensor.div_(world_size)

    # \overline{E_i} = \overline{E_{i-1}} + \overline{C[\nabla F_i - E_{i-1}]}
    if state.use_error_feedback == "ef21":
        state.global_error_dict[bucket_index].add_(input_tensor, alpha=1.0) # \overline{E_i} = \overline{E_{i-1}} + \overline{C[\nabla F_i - E_{i-1}]}
        input_tensor.copy_(state.global_error_dict[bucket_index])

    state.maybe_increase_iter(bucket)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor / world_size)
    return fut

# TODO: has async bug
def flex_quant_hook(
    state: FlexQuantState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor)

    # Get dtype and device.
    dtype, device = input_tensor.dtype, input_tensor.device
    original_shape = input_tensor.shape
    rank = dist.get_rank(group=group_to_use)

    # Compress the tensor and get the scale.
    quantized_tensor, scale = compress_flexible_nbits_by_bucket(input_tensor, bits=state.quantization_bits, scale_method=state.scale_method, bucket_size=state.bucket_size)

    # Allocate memory and all gather the scales and zeros.
    all_ranks_scale = _get_allgather_out_list(scale, world_size)

    # First, allgather scale and zeros.
    # dist.barrier()
    fut = dist.all_gather(
        all_ranks_scale, scale, group=group_to_use, async_op=True
    ).get_future()
    # dist.barrier()

    def allgather_quantized(fut):
        # dist.barrier()
        # Allgather quantized tensors.
        return (
            dist.all_gather(
                _get_allgather_out_list(quantized_tensor, world_size),
                quantized_tensor,
                group=group_to_use,
                async_op=True,
            ).get_future().wait()[0]
        )

    def dequantize_and_aggregate(fut):
        # dist.barrier()
        # Allgathered quantized tensors.
        all_ranks_quantized_tensor = fut.value()
        # print(type(all_ranks_quantized_tensor))
        # print(all_ranks_quantized_tensor.shape)

        # Zero the input tensor.
        # input_tensor.zero_()
        aggregated_tensor = torch.zeros(original_shape, dtype=dtype, device=device)

        # Using previously allgathered scales, dequantize gradient tensors locally and then aggregate them.

        for scale, quantized_tensor in zip(all_ranks_scale, all_ranks_quantized_tensor):
            dequantized_tensor = decompress_flexible_nbits_by_bucket(quantized_tensor, scale, bits=state.quantization_bits, original_shape=original_shape, bucket_size=state.bucket_size)


            # if bucket.is_last():
            #     print(f"Rank{rank} ,[{r}][After]Dequantized tensor: {dequantized_tensor}, dtype: {dequantized_tensor.dtype}", flush=True) # equal
            #     print("", end="", flush=True) # not equal
            #     print(dequantized_tensor, flush=True)   # equal
            #     dist.barrier() # equal
            dist.barrier() # equal

            aggregated_tensor.add_(dequantized_tensor)

        # if bucket.is_last():
        #     if state.iter < 120:
        #         # dist.barrier() # equal
        #         # 似乎整个hook是异步操作，中途观察的值可能是不一样的，但是在外部检查的gradient总是一样的
        #         state.util_dict[state.iter] = input_tensor
        #     elif state.iter == 120:
        #         torch.save(state.util_dict, f"agg_tensor_{rank}.pt")

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize(device)

            
        state.maybe_increase_iter(bucket)
        return aggregated_tensor / world_size

    return fut.then(allgather_quantized).then(dequantize_and_aggregate)

# TODO: has async bug
def layer_quant_hook(
    state: FlexQuantState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer
    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()

    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor)

    # Get dtype and device.
    dtype, device = input_tensor.dtype, input_tensor.device
    rank = dist.get_rank()

    # Unflatten the input tensor into per-parameter tensors, for layer-wise compression.
    tensors = bucket.gradients()
    
    # Collect the shapes of the input tensors.
    shapes = [tensor.shape for tensor in tensors]

    # Compute the total number of elements in the input tensors.
    int8_total_numel = input_tensor.numel() * state.quantization_bits // 8 + len(tensors) * state.bucket_size
    scale_total_numel = input_tensor.numel() // state.bucket_size + len(tensors)

    # Allocate memory for the quantized tensors and scales.
    int8_memory = torch.zeros(int8_total_numel, dtype=torch.uint8, device=device)
    scale_memory = torch.zeros(scale_total_numel, dtype=torch.float16, device=device)
    
    # Offset for the memory allocation.
    int8_offset, scale_offset = 0, 0
    int8_numels, scale_numels = [], []

    # Quantize the input tensors and store them in the allocated memory.
    for tensor in tensors:
        # Compress the tensor and get the scale.
        quantized_tensor, scale = compress_flexible_nbits_by_bucket(tensor, bits=state.quantization_bits, scale_method=state.scale_method, bucket_size=state.bucket_size)
        
        # Store the quantized tensor and scale in the allocated memory.
        int8_memory[int8_offset:int8_offset+quantized_tensor.numel()] = quantized_tensor.view(-1)
        scale_memory[scale_offset:scale_offset+scale.numel()] = scale.view(-1)
        
        # Store the number of elements in the quantized tensor and scale.
        int8_numels.append(quantized_tensor.numel())
        scale_numels.append(scale.numel())
        
        # Update the offset.
        int8_offset += quantized_tensor.numel()
        scale_offset += scale.numel()
        
        del quantized_tensor, scale

    # Allocate memory and all gather the scales and zeros.
    all_ranks_scale = _get_allgather_out_list(scale_memory, world_size)
    
    # First, allgather scale and zeros.
    fut = dist.all_gather(
        all_ranks_scale, scale_memory, group=group_to_use, async_op=True
    ).get_future()

    def allgather_quantized(fut):
        fut.wait()
        # Allgather quantized tensors.
        fut = dist.all_gather(
            _get_allgather_out_list(int8_memory, world_size),
            int8_memory,
            group=group_to_use,
            async_op=True,
        ).get_future()

        return fut.wait()

    def dequantize_and_aggregate(fut):  
        all_ranks_int8_memory = fut.wait()[0]

        # set input tensor to 0
        input_tensor.zero_()

        # Using previously allgathered scales, dequantize gradient tensors locally and then aggregate them.
        for recv_scale_memory, recv_int8_memory in zip(all_ranks_scale ,all_ranks_int8_memory):
            int8_offset, scale_offset = 0, 0
            # dequantize and aggregate
            for int8_numel, scale_numel, tensor, original_shape in zip(int8_numels, scale_numels, tensors, shapes):
                # index quantized tensor and scale
                quantized_tensor = recv_int8_memory[int8_offset:int8_offset+int8_numel]
                scale = recv_scale_memory[scale_offset:scale_offset+scale_numel]
                # dequantized tensor and add to input tensor
                dequantized_tensor = decompress_flexible_nbits_by_bucket(quantized_tensor, scale, bits=state.quantization_bits, original_shape=original_shape, bucket_size=state.bucket_size)
                tensor.add_(dequantized_tensor)
                # update offset
                int8_offset += int8_numel
                scale_offset += scale_numel

        return input_tensor / world_size

    return fut.then(allgather_quantized).then(dequantize_and_aggregate)

if __name__ == "__main__":
    # test quantizer on a tensor
    import torch
    torch.manual_seed(42)
    # get x
    x = torch.randn(2, 15)
    x = x.cuda()
    x[0, 1] = 14.3
    x[1, 1] = -10.6
    x = x + 2.0
    # get bits
    bits = 1
    # quantize
    x_quant, scale = compress_flexible_nbits_by_bucket(x, bits=bits, scale_method='max', bucket_size=8)
    print(x_quant)
    print(scale)
    x_dequant = decompress_flexible_nbits_by_bucket(x_quant, scale, bits=bits, original_shape=x.shape, bucket_size=8)
    print(x)
    print(x_dequant)
    
