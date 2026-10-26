"""Apply candidate workload mapping only to communication-method parameters."""
from argparse import Namespace

COMPRESSION_FIELDS = frozenset({
    'compressor', 'start_compress_iter', 'use_error_feedback', 'compress_rank',
    'min_compression_rate', 'update_proj_gap', 'quantization_bits', 'bucket_size',
    'scale_method', 'sparse_type', 'compress_ratio', 'scale', 'sigma_type',
    'beta_ef', 'error_inherit',
})


def apply_compression_policy(args, workload):
    from method_policy import configure_compression
    method = Namespace(**{key: value for key, value in vars(args).items() if key in COMPRESSION_FIELDS})
    configured = configure_compression(method, workload)
    if not isinstance(configured, Namespace) or set(vars(configured)) - COMPRESSION_FIELDS:
        raise ValueError('method policy may return only compression fields')
    for key, value in vars(configured).items():
        setattr(args, key, value)
    return args
