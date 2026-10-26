"""Candidate-owned frozen workload adaptation for compression settings.

Training budgets, data, and optimizer settings belong to the task contract.
Candidates may edit this policy and the shared communication-hook implementation
so the same submitted method can adapt its compression to each workload.
"""


def configure_compression(args, workload):
    """Apply stock GreedyLoRE vision defaults, preserving explicit overrides."""
    if workload != 'resnet18_cifar100_muon' or args.compressor != 'top_subspace':
        return args
    defaults = dict(compress_rank=64, start_compress_iter=4000, update_proj_gap=1200,
                    min_compression_rate=2., use_error_feedback='ef14', beta_ef=0., error_inherit=0.)
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    return args
