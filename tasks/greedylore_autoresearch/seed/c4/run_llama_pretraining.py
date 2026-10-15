import os
import time
import json
import math
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from c4.pept_utils import training_utils, args_utils
from c4.pept_utils.dataloader import PreprocessedIterableDataset
from c4.pept_utils.c4_data import load_c4_split
from c4.pept_utils.modeling_llama import LlamaForCausalLM

from comm_hooks.utils import add_comm_hook_args, get_run_name_c4, register_comm_hook_for_ddp_model
from optimizer import add_muon_args, build_muon_optimizer

transformers.logging.set_verbosity_error()


def set_model_pad_token_id(model, pad_token_id):
    module = model.module if hasattr(model, "module") else model
    module.config.pad_token_id = pad_token_id
    if module.generation_config is not None:
        module.generation_config.pad_token_id = pad_token_id


def require_training_budget(update_step, target_steps):
    if update_step < target_steps:
        raise RuntimeError(
            f"Training data exhausted after {update_step} of {target_steps} update steps"
        )


def training_target_steps(scheduler_steps, stop_after_steps):
    target = scheduler_steps if stop_after_steps is None else stop_after_steps
    if not 0 < target <= scheduler_steps:
        raise ValueError("stop_after_steps must be between 1 and num_training_steps")
    return target


def training_batches(dataloader, repeat=False, batch_size=None):
    while True:
        consumed = False
        for batch in dataloader:
            if batch_size is not None and batch['input_ids'].shape[0] != batch_size:
                continue
            consumed = True
            yield batch
        if not repeat:
            return
        if not consumed:
            raise RuntimeError("Cannot repeat an empty training dataloader")


def wandb_run_url(run):
    return run.url


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default='./c4/c4_en')
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--stop_after_steps", type=int, default=None,
                        help="Stop early while retaining the num_training_steps LR schedule.")
    parser.add_argument("--repeat_training_data", action="store_true",
                        help="Restart the existing dataloader when exhausted to finish the update budget.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", type=str, default="test")

    # Optimizer parameters
    parser.add_argument("--optimizer", default="Adam")

    # AdamW type parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clipping", type=float, default=0.0)   

    # Wandb log information
    parser.add_argument("--wandb_project", type=str, default="Compressed-AdamW-pretraining")
    parser.add_argument("--wandb_job_type", type=str, default="Llama-pretraining")
    # parser.add_argument("--wandb_run_name", type=str, default="AdamW-Llama-pretraining")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    # Compressor arguments
    from comm_hooks.utils import add_comm_hook_args
    add_comm_hook_args(parser)
    add_muon_args(parser, scalar_lr_default=0.001, scalar_weight_decay_default=0.0)
    
    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)

    supported_optimizers = ['adamw','galore_adamw','golore_adamw','muon']
    assert args.optimizer in supported_optimizers, "`optimizer` should be one of the following: " + ', '.join(supported_optimizers)

    return args


@torch.no_grad()
def evaluate_model(model, dataset_path, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    
    from requests.exceptions import ConnectionError
    for attempt in range(5):
        try:
            val_data = load_c4_split(dataset_path, split="validation")
        except ConnectionError as e:
                    if attempt < 5 - 1:
                        print(f"Connection error: {e}. Retrying...")
                        time.sleep(5)
                    else:
                        raise e
                    
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    loss_numerator = torch.tensor(0.0, dtype=torch.float64, device=device)
    local_tokens = torch.tensor(0, dtype=torch.int64, device=device)
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens >= target_eval_tokens:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        valid_tokens = (labels[:, 1:] != -100).sum()
        loss_numerator += loss.detach().double() * valid_tokens
        local_tokens += valid_tokens
        evaluated_on_tokens += valid_tokens.item() * world_size

    dist.all_reduce(loss_numerator, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_tokens, op=dist.ReduceOp.SUM)
    if local_tokens.item() == 0:
        raise RuntimeError("C4 validation produced no target tokens")
    return (loss_numerator / local_tokens).item(), local_tokens.item()

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

def main(args):
    target_steps = training_target_steps(args.num_training_steps, args.stop_after_steps)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later)
    if global_rank == 0:
        wandb.init(project=args.wandb_project, job_type=args.wandb_job_type, name=get_run_name_c4(args))

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    from requests.exceptions import ConnectionError
    for attempt in range(5):
        try:
            data = load_c4_split(args.dataset_path, split="train")
        except ConnectionError as e:
                    if attempt < 5 - 1:
                        print(f"Connection error: {e}. Retrying...")
                        time.sleep(5)
                    else:
                        raise e
                    
    seed_for_shuffle = 42 
    
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        # checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)

        from safetensors.torch import load_file

        checkpoint_path = os.path.join(args.continue_from, "model.safetensors")

        # Load the model weights from the safetensors file
        state_dict = load_file(checkpoint_path)

        # Load the state dictionary into the model
        model.load_state_dict(state_dict, strict=True)

        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=target_steps - update_step, desc="Update steps", ncols=80)
    
    if args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(args.beta1,args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "muon":
        optimizer = build_muon_optimizer(
            model, lr=args.lr, scalar_lr=args.muon_scalar_lr,
            mu=args.muon_mu, weight_decay=args.weight_decay,
            scalar_weight_decay=args.muon_scalar_weight_decay,
            scalar_betas=(args.muon_scalar_beta1, args.muon_scalar_beta2),
            scalar_epsilon=args.muon_scalar_eps,
            muon_epsilon=args.muon_epsilon,
            adjust_lr=None if args.muon_adjust_lr == "none" else args.muon_adjust_lr,
            compile_orthogonalization=args.muon_compile,
            distributed_orthogonalization=not args.muon_local_orthogonalization,
        )
    elif args.optimizer.lower() == "galore_adamw":
        from optimizer import GaLoreAdamW
        from glue.utils import get_galore_param_groups
        galore_param = get_galore_param_groups(model, weight_decay=args.weight_decay,rank=args.rank,
                                                    update_proj_gap=args.update_proj_gap,scale=args.scale,proj_type=args.proj_type)
        optimizer = GaLoreAdamW(galore_param, lr=args.lr, betas=(args.beta1,args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    scheduler = training_utils.get_scheculer(
        optimizer=optimizer,
        scheduler_type=args.scheduler,  # cosine
        num_training_steps=args.num_training_steps, # 55000
        warmup_steps=args.warmup_steps, # 5500
        min_lr_ratio=args.min_lr_ratio, # 0.1
    )

    if args.continue_from is not None:
        checkpoint_path = os.path.join(args.continue_from, 'optimizer.pt')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"Optimizer state loaded from {checkpoint_path}")

        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info(f"Scheduler state loaded from {checkpoint_path}")

    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        process_group = dist.distributed_c10d._get_default_group()
        register_comm_hook_for_ddp_model(model, process_group, args)

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    set_model_pad_token_id(model, tokenizer.pad_token_id)
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    best_eval_loss = float("inf")
    best_eval_step = None

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    torch.cuda.reset_peak_memory_stats()

    batches = training_batches(dataloader, args.repeat_training_data,
                               args.batch_size if args.repeat_training_data else None)
    for batch_idx, batch in enumerate(batches):

        if update_step >= target_steps:
            logger.info(f"Reached max number of update steps ({target_steps}). Stopping training.")
            print(f"Rank {global_rank} stopping training.")
            break

        global_step += 1
        local_step += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch, labels=labels).loss
        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        # The below code is only executed during the update step

        # add grad clipping
        if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        if global_rank == 0: pbar.update(1)
        
        optimizer.step()
        scheduler.step()

        # if global_rank == 0:
        #     grad_norm = 0.0
        #     for param in model.parameters():
        #         if param.grad is not None:
        #             grad_norm += param.grad.norm().item() ** 2
        #     grad_norm = torch.sqrt(torch.tensor(grad_norm)).item()

        optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
            save_dir_path = os.path.abspath(args.save_dir)
            os.makedirs(save_dir_path, exist_ok=True)
            # model.module.save_pretrained(current_model_directory, max_shard_size='100GB')

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            # torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            # with open(f"{current_model_directory}/training_state.json", "w") as f:
            #     json.dump(training_state_checkpoint, f, indent=4)
                
            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            # with open(f"{args.save_dir}/wandb.json", "w") as f:
            #     json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0 or update_step == 1:
            logger.info(f"Eval Every Step: {args.eval_every}")
            logger.info(f"Performing evaluation at step {update_step}")
            model.eval()
            total_loss, evaluated_on_tokens = evaluate_model(
                model, args.dataset_path, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
            )
            model.train()
            if total_loss < best_eval_loss:
                best_eval_loss = total_loss
                best_eval_step = update_step
            if global_rank == 0:
                wandb.log({
                    "eval_loss": total_loss,
                    "eval_ppl": math.exp(total_loss),
                    "eval_tokens": evaluated_on_tokens,
                    "best_eval_ppl": math.exp(best_eval_loss),
                    },
                    step=global_step,
                )
            logger.info(f"Eval loss at step {update_step}: {total_loss}")

        lr = optimizer.param_groups[0]["lr"]

        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        if global_rank == 0:

            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

            wandb.log({
                "loss": loss.item(),
                "lr": lr,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "throughput_tokens": tokens_in_update / update_time,
                "throughput_examples": args.total_batch_size / update_time,
                "throughput_batches": batches_in_update / update_time,
                "peak_memory_MB": peak_memory,
                # "grad_norm": grad_norm,
                },
                step=global_step,
            )
        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    require_training_budget(update_step, target_steps)
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    # current_model_directory = f"{args.save_dir}/model_{update_step}"
    # if global_rank == 0 and os.path.exists(current_model_directory):
    #     logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
    #     os.makedirs(args.save_dir, exist_ok=True)
    #     # model.module.save_pretrained(current_model_directory)

    #     optimizer_checkpoint = {
    #         "optimizer": optimizer.state_dict(),
    #         "scheduler": scheduler.state_dict(),
    #         "update_step": update_step,
    #         "global_step": global_step,
    #         "config": run_config,
    #         "wandb": wandb.run.dir,
    #         "dtype": args.dtype,
    #     }
    #     torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

    #     training_state_checkpoint = {
    #         "global_step": global_step,
    #         "update_step": update_step,
    #         "tokens_seen": tokens_seen,
    #         "tokens_seen_before": tokens_seen_before,
    #         "update_time": update_time,
    #     }
    #     with open(f"{current_model_directory}/training_state.json", "w") as f:
    #         json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model, args.dataset_path, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
    )
    if total_loss < best_eval_loss:
        best_eval_loss = total_loss
        best_eval_step = update_step

    if global_rank == 0:
        wandb.log({
            "final_eval_loss": total_loss,
            "final_eval_ppl": math.exp(total_loss),
            "final_eval_tokens": evaluated_on_tokens,
            "best_eval_ppl": math.exp(best_eval_loss),
            },
            step=global_step,
        )
        logger.info(f"Final eval loss: {total_loss}")
        
        if args.output_dir is not None :
            all_results = {
                "final_eval_loss": total_loss,
                "final_eval_ppl": math.exp(total_loss),
                "final_eval_tokens": evaluated_on_tokens,
                "best_eval_loss": best_eval_loss,
                "best_eval_ppl": math.exp(best_eval_loss),
                "best_eval_step": best_eval_step,
                "update_step": update_step,
                "wandb_link": wandb_run_url(wandb.run)
            }
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
