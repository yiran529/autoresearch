import torch
import torch.distributed as dist

import logging
from optimizer import PrecondAdam

logger = logging.getLogger(__name__)

def check_tensor_identity(tensor, group, desc="Tensor"):
    tensor_copy = tensor.clone().detach()
    dist.all_reduce(tensor_copy, group=group, op=dist.ReduceOp.AVG)
    if not torch.allclose(tensor, tensor_copy, atol=1e-6):
        raise RuntimeError(f"{desc} is not identical across all processes.")

def check_error_identity(error, global_error, group, iteration):
    error_average = torch.clone(error).detach()
    global_error_average = torch.clone(global_error).detach()

    dist.all_reduce(error_average, group=group, async_op=False)
    error_average.div_(group.size())

    dist.all_reduce(global_error_average, group=group, async_op=False)
    global_error_average.div_(group.size())

    if not torch.allclose(global_error_average, global_error, atol=1e-4):
        logger.info(f"Rank[{dist.get_rank()}] Iter[{iteration}], Last 5 elements of Global Error Average: {global_error_average[-5:]}, Last 5 elements of Global Error: {global_error[-5:]}")
        raise RuntimeError(f"Iter[{iteration}]Global Error feedback is not consistent.")

    if not torch.allclose(error_average, global_error, atol=1e-4):
        logger.info(f"Rank[{dist.get_rank()}] Iter[{iteration}], Last 5 elements of Local Error: {error[-5:]}, Last 5 elements of Local Error Average: {error_average[-5:]}, Last 5 elements of Global Error: {global_error[-5:]}")
        raise RuntimeError(f"Iter[{iteration}]Local Error feedback is not consistent. Difference: {torch.norm(error_average - global_error)}. Different elements count: {torch.sum(error_average != global_error)}/{error_average.numel()}")


def _get_allgather_out_list(all_gather_in_list, world_size):
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list

def dtype_bits(tensor):
    dtype = tensor.dtype
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits
    elif dtype.is_complex:
        return torch.finfo(dtype).bits * 2  # Complex numbers have twice the bits
    elif dtype == torch.bool:
        return 1
    elif "int" in str(dtype):
        return torch.iinfo(dtype).bits
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
def tensor_bits(tensor):
    return tensor.numel() * dtype_bits(tensor)
    
class HookState:
    
    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group
        self.start_compress_iter = 0
        self.iter = 0

        self.total_bit_before_compression = 0
        self.total_bit_after_compression = 0
        self.compressor_name="none compressor"

        self.compress_momentum = False
        self.param_state = None
        self.param_to_name = None
        self.beta1 = None
        self.adam_freeze_key = False

    def init_momentum_field(self, param_state, beta1):
        self.param_state = param_state
        self.beta1 = beta1
        self.compress_momentum = True

    def maybe_accumulate_momentum_on_bucket(self, bucket: dist.GradBucket):
        if not self.compress_momentum: 
            return
        if self.iter >= self.start_compress_iter and not self.adam_freeze_key:
            self.adam_freeze_key = True
            logger.info(f"Freeze the second momentum of Adam optimizer after {self.iter}(included) steps")
        if self.adam_freeze_key:
            self.accumulate_momentum_on_bucket(bucket)

    def accumulate_momentum_on_bucket(self, bucket: dist.GradBucket):
        if not self.compress_momentum:
            raise RuntimeError("Momentum compression is not enabled!")
        if self.param_state is None:
            raise RuntimeError("Parameter state is not initialized!")
        # accumulate momentum
        parameters, gradients = bucket.parameters(), bucket.gradients()
        assert len(parameters) == len(gradients), "The number of parameters and gradients should be the same."
        for tensor, grad in zip(parameters, gradients):
            state = self.param_state[tensor]
            grad.mul_(1 - self.beta1).add_(state['exp_avg'], alpha=self.beta1)


    def maybe_increase_iter(self, bucket):
        """Track iterations and trigger log message at start of local SGD."""
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last():
            self.iter += 1

            if self.iter == self.start_compress_iter:
                logger.info(f"Start to apply {self.compressor_name} hook after {self.start_compress_iter} iterations.")

    def compression_bits_stats(self):

        compress_rate = (
            self.total_bit_before_compression / self.total_bit_after_compression
            if self.total_bit_after_compression > 0
            else 0
        )
        return (
            compress_rate,
            self.total_bit_before_compression,
            self.total_bit_after_compression,
        )

def get_uncompress_names(ddp_model):
    assert hasattr(ddp_model, "module"), "The model should be wrapped by DDP."
    assert isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel), "The model should be wrapped by DDP."
    model_name = ddp_model.module.__class__.__name__.lower()
    if "llama" in model_name:
        return ["lm_head", "embed_tokens"]
    elif "roberta" in model_name:
        return ["classifier", "embedding"]

def register_comm_hook_for_ddp_model(model, process_group, args, optimizer=None):
    hook_state = None
    if args.compressor == 'powersgd' :
        from comm_hooks.powerSGD_hook import PowerSGDState, powerSGD_hook
        hook_state = PowerSGDState(
            process_group=process_group,
            matrix_approximation_rank=args.compress_rank,
            use_error_feedback=args.use_error_feedback,
            start_compress_iter=args.start_compress_iter,
            random_seed=args.seed,
        )
        model.register_comm_hook(hook_state, powerSGD_hook)
    elif args.compressor == 'original_powersgd':
        from comm_hooks.original_powerSGD_hook import OriginalPowerSGDState, original_powerSGD_hook
        use_ef_bool = args.use_error_feedback == "ef14"
        hook_state = OriginalPowerSGDState(
            process_group=process_group,
            matrix_approximation_rank=args.compress_rank,
            use_error_feedback=use_ef_bool,
            start_powerSGD_iter=args.start_compress_iter,
            random_seed=args.seed,
        )
        model.register_comm_hook(hook_state, original_powerSGD_hook)
    elif args.compressor == "lore":
        from comm_hooks.lore_hook import LoreState, lore_hook
        hook_state = LoreState(
            process_group=process_group,
            matrix_approximation_rank=args.compress_rank,
            update_proj_gap=args.update_proj_gap,
            use_error_feedback=args.use_error_feedback,
            start_compress_iter=args.start_compress_iter,
            uncompressed_names=get_uncompress_names(model),
            scale=args.scale,
            random_seed=args.seed,
        )
        model.register_comm_hook(hook_state, lore_hook)
    elif args.compressor in ["top_subspace", "top_rc", "rand_subspace", "rand_rc"]:
        from comm_hooks.subspace_hook import SubspaceState, subspace_hook
        random = 'rand' in args.compressor
        identity_proj = 'rc' in args.compressor
        hook_state = SubspaceState(
            process_group=process_group,
            matrix_approximation_rank=args.compress_rank,
            min_compression_rate=args.min_compression_rate,
            update_proj_gap=args.update_proj_gap,
            use_error_feedback=args.use_error_feedback,
            start_compress_iter=args.start_compress_iter,
            uncompressed_names=get_uncompress_names(model),
            random_seed=args.seed,
            identity_proj=identity_proj,
            scale=args.scale,
            random=random,
            sigma_type=args.sigma_type,
            beta_ef=args.beta_ef,
            error_inherit=args.error_inherit,
        )
        model.register_comm_hook(hook_state, subspace_hook)
    
    elif args.compressor in ["fake_top_subspace", "fake_top_rc", "fake_rand_subspace", "fake_rand_rc"]:
        from comm_hooks.fake_subspace_hook import FakeSubspaceState, fake_subspace_hook
        random = 'rand' in args.compressor
        identity_proj = 'rc' in args.compressor
        hook_state = FakeSubspaceState(
            process_group=process_group,
            matrix_approximation_rank=args.compress_rank,
            update_proj_gap=args.update_proj_gap,
            use_error_feedback=args.use_error_feedback,
            start_compress_iter=args.start_compress_iter,
            uncompressed_names=get_uncompress_names(model),
            random_seed=args.seed,
            identity_proj=identity_proj,
            scale=args.scale,
            random=random,
            sigma_type=args.sigma_type,
            beta_ef=args.beta_ef,
            error_inherit=args.error_inherit,
        )
        model.register_comm_hook(hook_state, fake_subspace_hook)
    elif args.compressor == "fake_powersgd":
        from comm_hooks.fake_powersgd_hook import FakePowerSGDState, fake_PowerSGD_hook
        hook_state = FakePowerSGDState(
            process_group=process_group,
            matrix_approximation_rank=args.compress_rank,
            use_error_feedback=args.use_error_feedback,
            start_compress_iter=args.start_compress_iter,
            uncompressed_names=get_uncompress_names(model),
            random_seed=args.seed,
            scale=args.scale,
            sigma_type=args.sigma_type,
            beta_ef=args.beta_ef,
            error_inherit=args.error_inherit,
        )
        model.register_comm_hook(hook_state, fake_PowerSGD_hook)
    elif args.compressor == "flex_quant_sync":
        from comm_hooks.flex_quant_hooks import flex_quant_hook_sync, FlexQuantState
        hook_state = FlexQuantState(
            process_group=process_group,
            quantization_bits=args.quantization_bits,
            bucket_size=args.bucket_size,
            scale_method=args.scale_method,
            use_error_feedback=args.use_error_feedback,
            start_compress_iter=args.start_compress_iter
        )
        model.register_comm_hook(hook_state, flex_quant_hook_sync)
    elif args.compressor == 'topk_sync' or args.compressor == 'randk_sync':
        from comm_hooks.sparse_hook import SparseState, sparse_hook_sync
        random = 'randk' in args.compressor
        hook_state = SparseState(
            process_group=process_group,
            compress_ratio=args.compress_ratio,
            sparse_type=args.sparse_type,
            use_error_feedback=args.use_error_feedback,
            random=random,
            start_compress_iter=args.start_compress_iter,
            random_seed=args.seed,
        )
        model.register_comm_hook(hook_state, sparse_hook_sync)
    elif args.compressor == 'noop':
        from comm_hooks.debugging_hooks import noop_hook
        model.register_comm_hook(None, noop_hook)
    elif args.compressor == 'none' :
        from comm_hooks.default_hooks import allreduce_hook
        model.register_comm_hook(process_group, allreduce_hook)
    else:
        raise ValueError(f"Compressor {args.compressor} not supported.")
    
    # For selective compression
    if hasattr(hook_state, 'param_to_name'):
        hook_state.param_to_name = {param: name for name, param in model.named_parameters()}
    # for param, name in hook_state.param_to_name.items():
    #     if dist.get_rank() == 0:
    #         logger.info(f"Parameter name: {name}, shape: {param.shape}")

    # For PrecondAdam
    if hook_state is not None and args.optimizer == 'precond_adam':
        from accelerate.optimizer import AcceleratedOptimizer
        if isinstance(optimizer, AcceleratedOptimizer):
            optimizer = optimizer.optimizer
        assert isinstance(optimizer, PrecondAdam), f"Get Optimizer type {type(optimizer)}, Optimizer should be PrecondAdam when using precond_adam."
        hook_state.init_momentum_field(optimizer.state, args.beta1)
        optimizer.hook_state = hook_state

def add_comm_hook_args(parser):
    ### Commpressor arguments
    parser.add_argument(
        "--compressor",
        type=str,
        default="none",
        help="Set the compressor to use.",
    )
    parser.add_argument(
        "--start_compress_iter",
        type=int,
        default=10,
        help="Set the iteration to start compression.",
    )
    parser.add_argument(
        "--use_error_feedback",
        type=str,
        default="noef",
        choices=["noef", "ef14", "ef21"],
        help="Set the error feedback to use.",
    )
    ### Specific arguments for each compressor
    parser.add_argument(
        "--compress_rank",
        type=int,
        default=4,
        help="Set the rank of the low-rank approximation.",
    )
    parser.add_argument(
        "--min_compression_rate", type=float, default=2.0,
        help="Compress a tensor when dense size / transmitted size exceeds this value.",
    )
    parser.add_argument(
        "--update_proj_gap",
        type=int,
        default=200,
        help="Set the gap between two projection updates.",
    )

    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=8,
        help="Set the number of bits used for quantization.",
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=512,
        help="Set the size of the bucket used for quantization.",
    )
    parser.add_argument(
        "--scale_method",
        type=str,
        default='max',
        help="Set the method used to calculate the scale for quantization.",
    )

    parser.add_argument(
        "--sparse_type",
        type=str,
        default='tensor',
        choices=['row', 'column', 'tensor'],
        help="Set the type of top-k sparsification to use.",
    )
    parser.add_argument(
        "--compress_ratio",
        type=float,
        default=0.08,
        help="Set the ratio of the top-k elements to keep.",
    )
    ### check whether the gradients are identical across all processes
    parser.add_argument(
        "--check_grad",
        action="store_true",
        default=False,
        help="Whether to check the identity of the gradients.",
    )

    ### Scale for any lore method
    parser.add_argument("--scale", type=float, default=1.0, help='galore_scale') 

    ### Galore and Golore arguments 
    parser.add_argument("--rank", type=int, default=16, help='galore_rank') 
    parser.add_argument("--proj_type", type=str, default="std", help='proj_type') 
    parser.add_argument("--rand_ratio", type=float, default=2, help='Golore rand_ratio')

    ### GreedyLore arguments
    parser.add_argument(
        "--sigma_type",
        type=int,
        default=0,
        help="Set the type of sigma used for top_subspace compressor.",
    )
    parser.add_argument("--beta_ef", type=float, default=0.0)
    parser.add_argument("--error_inherit", type=int, default=0)

    parser.add_argument("--wandb_run_name", type=str, default="")



def get_run_name_glue(args, run_name=''):
    run_name += f"{args.task_name}_{args.model_name_or_path.split('/')[-1]}"
    run_name += f"-dist{dist.get_world_size()}"
    run_name += f"-bs{args.per_device_train_batch_size}"
    run_name += f"-lr{args.learning_rate}"
    run_name += f"-wd{args.weight_decay}"
    run_name += f"-{args.optimizer}"
    run_name += f"-{args.compressor}"
    run_name += f"-{args.use_error_feedback}"
    if args.compressor in ['powersgd', 'lore'] :
        run_name += f"(rk{args.compress_rank})"
    elif 'topk' in args.compressor:
        run_name += f"(topk{args.compress_ratio})"
    elif 'flex_quant' in args.compressor:
        run_name += f"({args.quantization_bits}bit)"

    # time
    import datetime
    run_name += f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    return run_name

def get_run_name_c4(args, run_name=''):
    if args.wandb_run_name:
        return args.wandb_run_name
    if not run_name:
        run_name = ''
    run_name += f"c4_{args.model_config.split('/')[-1].split('.')[0]}"
    run_name += f"-dist{dist.get_world_size()}"
    run_name += f"-bs{args.batch_size}"
    run_name += f"-lr{args.lr}"
    run_name += f"-wd{args.weight_decay}"
    run_name += f"-gc{args.grad_clipping}"
    run_name += f"-{args.optimizer}"
    run_name += f"-{args.compressor}"
    run_name += f"-{args.use_error_feedback}"
    if args.compressor in ['powersgd', 'lore'] :
        run_name += f"(rk{args.compress_rank})"
    elif 'topk' in args.compressor:
        run_name += f"(topk{args.compress_ratio})"
    elif 'flex_quant' in args.compressor:
        run_name += f"({args.quantization_bits}bit)"
    
    # time
    import datetime
    run_name += f"-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    return run_name

if __name__ == '__main__':
    # test dtype_bits of fp, bf, int8, bool
    tensor = torch.randn(10, 10)
    print(f"float32 tensor dtype bits: {dtype_bits(tensor)}")
    tensor = torch.randn(10, 10).half()
    print(f"float16 tensor dtype bits: {dtype_bits(tensor)}")
    tensor = torch.randn(10, 10).bfloat16()
    print(f"bfloat16 tensor dtype bits: {dtype_bits(tensor)}")
    tensor = torch.randn(10, 10).to(torch.int8)
    print(f"int8 tensor dtype bits: {dtype_bits(tensor)}")
    tensor = torch.randn(10, 10).bool()
    print(f"bool tensor dtype bits: {dtype_bits(tensor)}")

    # test tensor_bits
    tensor = torch.randn(10, 10)
    print(f"float32 tensor bits: {tensor_bits(tensor)}")
    tensor = torch.randn(10, 10).half()
    print(f"float16 tensor bits: {tensor_bits(tensor)}")
    tensor = torch.randn(10, 10).bfloat16()
    print(f"bfloat16 tensor bits: {tensor_bits(tensor)}")
    tensor = torch.randn(10, 10).to(torch.int8)
    print(f"int8 tensor bits: {tensor_bits(tensor)}")
    tensor = torch.randn(10, 10).bool()
    print(f"bool tensor bits: {tensor_bits(tensor)}")

    # test tensor_bits of complex
    tensor = torch.randn(10, 10).to(torch.complex64)
    print(f"complex64 tensor bits: {tensor_bits(tensor)}")
    tensor = torch.randn(10, 10).to(torch.complex128)
    print(f"complex128 tensor bits: {tensor_bits(tensor)}")

    # test tensor_bits of quantized
    tensor = torch.randn(10, 10).to(torch.int8)
    print(f"int8 tensor bits: {tensor_bits(tensor)}")
    tensor = torch.randn(10, 10).to(torch.int16)
    print(f"int16 tensor bits: {tensor_bits(tensor)}")
    tensor = torch.randn(10, 10).to(torch.int32)
    print(f"int32 tensor bits: {tensor_bits(tensor)}")


    


