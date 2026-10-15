from typing import List

def get_default_param_groups(model, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "names": [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "names": [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def get_onebit_param_groups(model, weight_decay=0.01):
        # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'names': [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        }, 
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'names': [n for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    return optimizer_grouped_parameters

def get_galore_param_groups(
    model, weight_decay, rank=256, update_proj_gap=200, scale=0.25, proj_type="std"
) -> List[dict]:
    """
    It's advised to use this instead of manually specifying which param groups
    to apply GaLore on.
    """
    galore_params = []
    non_galore = []
    no_decay_params = []
    no_decay = ["bias", "LayerNorm.weight"]

    for name, param in model.named_parameters():
        # Only make sense to do SVD on 2d gradient matrices
        # e.g. nn.Linear, VocabEmbedding, etc.
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        elif param.dim() == 2 and any(target_key in name for target_key in ["attn", "attention", "mlp"]):
            galore_params.append(param)
        else:
            non_galore.append(param)

    param_groups = [
        {
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
            "weight_decay": weight_decay,
        },
        {"params": non_galore, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return param_groups