# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import cupy
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import time
import numpy as np

import torch.distributed as dist
from packaging import version as pkg_version

def torch2cupy(tensor):
    return cupy.fromDlpack(to_dlpack(tensor))

def cupy2torch(cupy_tensor):
    return from_dlpack(cupy_tensor.toDlpack())

def compress_by_chunk(cupy_bool_tensor, num_chunks):
    packed_sign = cupy.packbits(cupy_bool_tensor)
    sign_list_packed = cupy.split(packed_sign, num_chunks)
    cupy.cuda.get_current_stream().synchronize()
    return sign_list_packed

def required_torch_version(min_version=None, max_version=None):
    assert min_version or max_version, "Must provide a min_version or max_version argument"

    torch_version = pkg_version.parse(torch.__version__)

    if min_version and pkg_version.parse(str(min_version)) > torch_version:
        return False

    if max_version and pkg_version.parse(str(max_version)) < torch_version:
        return False

    return True

def device_name(device_index=None):
    if device_index is None:
        return 'cuda'
    return 'cuda:{}'.format(device_index)

def compressed_allreduce_old(buffer_m: torch.tensor, worker_error, server_error, local_rank):

    # params
    bool_not_supported = required_torch_version(min_version=1.10)
    world_group = dist.group.WORLD
    world_size = dist.get_world_size()
    self_rank = dist.get_rank() # Question: difference between self_rank and local_rank?

    # all_start_time = time.time()
    original_shape = buffer_m.size()
    if len(original_shape) > 1:
        buffer_m = torch.flatten(buffer_m)
    original_size = buffer_m.numel()
    worker_error_size = worker_error.numel()
    cupy.cuda.Device(local_rank).use()

    if original_size != worker_error_size:
        empty_tensor = torch.zeros(worker_error_size - original_size, device=buffer_m.device)
        buffer_m = torch.cat([buffer_m, empty_tensor])

    buffer_m.add_(worker_error)
    worker_scale = torch.linalg.norm(buffer_m) / np.sqrt(buffer_m.numel())
    worker_error.set_(buffer_m - worker_scale * buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

    if bool_not_supported:
        cupy_sign_list_packed = compress_by_chunk(
            torch2cupy(buffer_m.sign_().add_(1).bool().to(dtype=torch.uint8)), world_size)
    else:
        cupy_sign_list_packed = compress_by_chunk(
            torch2cupy(buffer_m.sign_().add_(1).bool()), world_size)
    cupy_worker_scale = torch2cupy(worker_scale)

    cupy_recvbuf_sign = cupy.zeros([world_size, cupy_sign_list_packed[self_rank].size],
                                    dtype=cupy_sign_list_packed[0].dtype)
    # cupy_recvbuf_scale = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)

    sign_list_packed = [
        cupy2torch(cupy_sign_list_packed[idx]) for idx in range(world_size)
    ]

    # worker_scale = cupy2torch(cupy_worker_scale)
    recvbuf_sign = cupy2torch(cupy_recvbuf_sign)
    #recvbuf_scale = cupy2torch(cupy_recvbuf_scale)
    recvbuf_scale = [
        torch.zeros(1, dtype=worker_scale.dtype, device=torch.device(device_name(local_rank)))
        for i in range(world_size)
    ]

    # communication phase 1
    # gather_start = time.time()
    # Alltoall for sign
    dist.all_to_all_single(recvbuf_sign, torch.stack(sign_list_packed), group=world_group)
    # Allgather for scale
    dist.all_gather(recvbuf_scale, worker_scale, group=world_group)

    # gather_end = time.time()

    # cupy_sign_list_packed, sign_list_packed, cupy_worker_scale, worker_scale = None, None, None, None
    cupy_sign_list_packed = None

    cupy_recvbuf_sign = torch2cupy(recvbuf_sign)
    #cupy_recvbuf_scale = torch2cupy(torch.stack(recvbuf_scale))

    compensated_server_m = cupy2torch(
        (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(world_size, -1)).float().add_(-0.5).mul_(2.0).mul_(
            torch.stack(recvbuf_scale).mul_(1 / world_size)).sum(0)
    compensated_server_m.add_(server_error)
    server_scale = torch.linalg.norm(compensated_server_m) / np.sqrt(compensated_server_m.numel())
    server_error.set_(compensated_server_m -
                        server_scale * compensated_server_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0))

    # cupy_server_scale = torch2cupy(server_scale)

    if bool_not_supported:
        cupy_server_sign_packed = compress_by_chunk(
            torch2cupy(compensated_server_m.sign_().add_(1).bool().to(dtype=torch.uint8)),
            1)
    else:
        cupy_server_sign_packed = compress_by_chunk(
            torch2cupy(compensated_server_m.sign_().add_(1).bool()), 1)
    compensated_server_m = None

    cupy_recvbuf_sign_server = cupy.zeros([world_size, cupy_server_sign_packed[0].size],
                                            dtype=cupy_recvbuf_sign.dtype)
    # cupy_recvbuf_sign, recvbuf_sign = None, None
    cupy_recvbuf_sign = None

    server_sign_packed = [cupy2torch(cupy_server_sign_packed[0])]
    recvbuf_sign_server = [
        cupy2torch(cupy_recvbuf_sign_server[idx]) for idx in range(world_size)
    ]

    # server_scale = cupy2torch(cupy_server_scale)
    cupy_recvbuf_scale_server = cupy.zeros([world_size, 1], dtype=cupy_worker_scale.dtype)
    # cupy_recvbuf_scale, recvbuf_scale = None, None

    recvbuf_scale_server = [
        cupy2torch(cupy_recvbuf_scale_server[idx]) for idx in range(world_size)
    ]

    # Communication Phase 2
    dist.all_gather(recvbuf_sign_server, server_sign_packed[0], group=world_group)
    dist.all_gather(recvbuf_scale_server, server_scale, group=world_group)

    cupy_server_sign_packed = None

    # need to convert from a tensor list to a single tensor
    # dist.all_gather only provides a tensor list as the recv/output buffer
    recvbuf_sign_server = torch.stack(recvbuf_sign_server)

    cupy_recvbuf_sign_server = torch2cupy(recvbuf_sign_server)

    buffer_m.data.copy_(
        cupy2torch((cupy.unpackbits(cupy_recvbuf_sign_server.flatten())).reshape(
            world_size, -1)).float().add_(-0.5).mul_(2.0).mul_(
                cupy2torch(cupy_recvbuf_scale_server)).flatten().data)
    if original_size != worker_error_size:
        buffer_m = buffer_m[0:original_size]
    if len(original_shape) > 1:
        buffer_m = buffer_m.reshape(original_shape)

    return buffer_m


def compressed_allreduce(buffer_m: torch.tensor, worker_error, server_error, local_rank):

    # params
    bool_not_supported = required_torch_version(min_version=1.10)
    world_group = dist.group.WORLD
    world_size = dist.get_world_size()
    self_rank = dist.get_rank() # Question: difference between self_rank and local_rank?

    # all_start_time = time.time()
    original_shape = buffer_m.size()
    if len(original_shape) > 1:
        buffer_m = torch.flatten(buffer_m)
    original_size = buffer_m.numel()
    worker_error_size = worker_error.numel()

    if original_size != worker_error_size:
        empty_tensor = torch.zeros(worker_error_size - original_size, device=buffer_m.device)
        buffer_m = torch.cat([buffer_m, empty_tensor])

    buffer_m.add_(worker_error)
    # cal C[m]
    worker_scale = torch.linalg.norm(buffer_m) / np.sqrt(buffer_m.numel())
    compressed_buffer_m = worker_scale * buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
    # cal delta = C[m] - m
    worker_error.set_(buffer_m - compressed_buffer_m)
    # average
    dist.all_reduce(compressed_buffer_m, group=world_group)
    buffer_m.set_(compressed_buffer_m.div_(world_size))

    if original_size != worker_error_size:
        buffer_m = buffer_m[0:original_size]
    if len(original_shape) > 1:
        buffer_m = buffer_m.reshape(original_shape)

    return buffer_m


def compressed_allreduce_bucket(buffer_m: torch.tensor, worker_error, server_error, local_rank, bucket_size=256):

    # params
    bool_not_supported = required_torch_version(min_version=1.10)
    world_group = dist.group.WORLD
    world_size = dist.get_world_size()
    self_rank = dist.get_rank() # Question: difference between self_rank and local_rank?

    # all_start_time = time.time()
    original_shape = buffer_m.size()
    if len(original_shape) > 1:
        buffer_m = torch.flatten(buffer_m)
    original_size = buffer_m.numel()
    worker_error_size = worker_error.numel()

    if original_size != worker_error_size:
        empty_tensor = torch.zeros(worker_error_size - original_size, device=buffer_m.device)
        buffer_m = torch.cat([buffer_m, empty_tensor])

    buffer_m.add_(worker_error)
    if worker_error_size % bucket_size != 0:
        empty_tensor = torch.zeros(bucket_size - worker_error_size % bucket_size, device=buffer_m.device)
        buffer_m = torch.cat([buffer_m, empty_tensor])

    # cal C[m]
    buffer_m = buffer_m.view(-1, bucket_size)
    worker_scale = torch.linalg.norm(buffer_m, dim=1) / np.sqrt(bucket_size)
    worker_scale = worker_scale.unsqueeze(1)
    compressed_buffer_m = worker_scale * buffer_m.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
    # cal delta = C[m] - m
    worker_error.set_((buffer_m - compressed_buffer_m).view(-1)[:worker_error_size])
    # average C[m]
    dist.all_reduce(compressed_buffer_m, group=world_group)
    buffer_m = buffer_m.view(-1)
    buffer_m = compressed_buffer_m.div_(world_size).reshape(-1)

    if original_size != worker_error_size:
        buffer_m = buffer_m[0:original_size]
    if len(original_shape) > 1:
        buffer_m = buffer_m.reshape(original_shape)

    return buffer_m
