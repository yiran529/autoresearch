# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning a 🤗 Transformers model for sequence classification on GLUE."""

import argparse
import json
import logging
import math
import os
import random

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import torch.distributed as dist
from utils import get_default_param_groups, get_galore_param_groups, get_onebit_param_groups
from comm_hooks.utils import register_comm_hook_for_ddp_model, get_run_name_glue
from optimizer import add_muon_args, build_muon_optimizer
from glue.local_glue import load_local_glue

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.44.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def check_grad_identity(model):
    """
    检查所有 DDP 不同机器上的模型对应的梯度是否相同。
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 获取当前进程的梯度
            grad = param.grad.clone()

            # 使用 all_reduce 聚合所有进程的梯度
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)

            # 计算平均梯度
            grad_mean = grad / dist.get_world_size()

            # 检查当前进程的梯度是否与平均梯度相同
            if not torch.allclose(param.grad, grad_mean, atol=1e-6):
                logger.info(f"Gradient mismatch in parameter: {name}")
                return False

    # print("All gradients are identical across all processes.")
    return True
    

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument("--local_glue_cache_root", type=str, default=None,
                        help="Read prepared GLUE Arrow splits from this local cache root.")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_fraction", type=float, default=0.0, help="Fraction of steps to warmup the learning rate for."
    )
    # parser.add_argument(
    #     "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    # )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )


    ###Custom arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Set run project for wandb log.",
    )
    parser.add_argument(
        "--wandb_job_type",
        type=str,
        default=None,
        help="Set run job_type for wandb log.",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default='adamw',
        help="Set optimizer to use.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Set coefficient used for computing running averages of gradient.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Set coefficient used for computing running averages of sqare gradient.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Set term added to the denominator to improve numerical stability.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='fp16',
        help="Set dtype for optimizer.",
    )



    # Compressor arguments
    from comm_hooks.utils import add_comm_hook_args
    add_comm_hook_args(parser)
    add_muon_args(parser)

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    # Custom Sanity Checks
    if args.wandb_project is not None or args.wandb_job_type is not None or args.wandb_run_name is not None:
        assert args.report_to == 'wandb', "Need to set report_to to wandb to use wandb specific arguments."
    
    supported_optimizers = ['adamw', 'sgd', 'precond_adam', 'onebit_adam', 'onebit_adammini', 'galore', 'muon']
    assert args.optimizer in supported_optimizers, "`optimizer` should be one of the following: " + ', '.join(supported_optimizers)

    return args

def main():
    args = parse_args()
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    if args.dtype != "fp16":
        accelerator = (
            Accelerator(log_with=args.report_to, project_dir=args.output_dir,
                        step_scheduler_with_optimizer=False)
            if args.with_tracking else Accelerator(step_scheduler_with_optimizer=False)
        )
    else:
        accelerator = (
            Accelerator(mixed_precision="fp16", log_with=args.report_to,
                        project_dir=args.output_dir, step_scheduler_with_optimizer=False)
            if args.with_tracking else Accelerator(mixed_precision="fp16", step_scheduler_with_optimizer=False)
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = (load_local_glue(args.local_glue_cache_root, args.task_name)
                        if args.local_glue_cache_root else load_dataset("nyu-mll/glue", args.task_name))
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: "
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        
        ### data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.mixed_precision == 'fp16' else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    if args.optimizer == 'adamw' :
        optimizer_grouped_parameters = get_default_param_groups(model, args.weight_decay)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1,args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'muon' :
        optimizer = build_muon_optimizer(
            model, lr=args.learning_rate, scalar_lr=args.muon_scalar_lr,
            mu=args.muon_mu, weight_decay=args.weight_decay,
            scalar_weight_decay=args.muon_scalar_weight_decay,
            scalar_betas=(args.muon_scalar_beta1, args.muon_scalar_beta2),
            scalar_epsilon=args.muon_scalar_eps,
            muon_epsilon=args.muon_epsilon,
            adjust_lr=None if args.muon_adjust_lr == "none" else args.muon_adjust_lr,
            compile_orthogonalization=args.muon_compile,
            distributed_orthogonalization=not args.muon_local_orthogonalization,
        )
    elif args.optimizer == 'sgd' :
        optimizer_grouped_parameters = get_default_param_groups(model, args.weight_decay)
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'precond_adam' :
        from optimizer.precond_adam import PrecondAdam
        optimizer_grouped_parameters = get_default_param_groups(model, args.weight_decay)
        optimizer = PrecondAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1,args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'galore' :
        from optimizer.galore_adamw import AdamW as GaloreAdamW
        optimizer_grouped_parameters = get_galore_param_groups(
            model=model, 
            weight_decay=args.weight_decay,
            rank=args.compress_rank,
            update_proj_gap=args.update_proj_gap,
            scale=args.scale,
            proj_type='std',
        )
        optimizer = GaloreAdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.beta1,args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == 'onebit_adam' :
        from optimizer.onebit_adam import OneBitAdam
        optimizer_grouped_parameters = get_onebit_param_groups(model, args.weight_decay)
        optimizer = OneBitAdam(optimizer_grouped_parameters, lr=args.learning_rate, freeze_step=args.start_compress_iter, betas=(args.beta1,args.beta2), eps=args.eps)
    elif args.optimizer == 'onebit_adammini' :
        from optimizer.onebit_adammini import OneBitAdamMini
        optimizer_grouped_parameters = get_onebit_param_groups(model, args.weight_decay)
        dim = model.config.hidden_size
        n_heads = model.config.num_attention_heads
        optimizer = OneBitAdamMini(optimizer_grouped_parameters, lr=args.learning_rate, freeze_step=args.start_compress_iter, betas=(args.beta1,args.beta2), eps=args.eps, dim=dim, n_heads=n_heads)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")
    
    # Prepare data before calculating the per-rank update budget.
    overrode_max_train_steps = args.max_train_steps is None
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    logger.info(str(model))
    logger.info(str(type(model)))



    # Accelerate shards the train loader across ranks; schedule each optimizer update once.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_fraction * args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Compressor
    process_group = dist.distributed_c10d._get_default_group()
    register_comm_hook_for_ddp_model(model, process_group, args, optimizer=optimizer)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value

        if args.report_to == 'wandb' :
            wandb_run_name = args.wandb_run_name or get_run_name_glue(args)
            accelerator.init_trackers(args.wandb_project, experiment_config, init_kwargs={"wandb":{"name":wandb_run_name, "job_type":args.wandb_job_type}})
        else :
            accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    torch.cuda.reset_peak_memory_stats()

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    best_metric = None

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                optimizer.step()
                if args.check_grad:
                    check_grad_identity(model)
                lr_scheduler.step()
                optimizer.zero_grad() 
                progress_bar.update(1)
                completed_steps += 1

                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

                if args.with_tracking :
                    accelerator.log(
                        {
                            "loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "weight_decay": optimizer.param_groups[0]["weight_decay"],
                            "peak_memory_MB": peak_memory,
                        },
                        step=completed_steps,
                    )
                logger.info(f"step {completed_steps}, loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}, peak_memory_MB: {peak_memory}")

                # nan
                if loss.item() == float('inf') or loss.item() == float('-inf') or math.isnan(loss.item()):
                    logger.info(f"Got an inf or nan loss, ending training. Try lowering the learning rate.")
                    return
                

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        logger.info(f"epoch {epoch}, eval metric: {eval_metric}")
        
        if not best_metric:
            best_metric = eval_metric
        else:
            for key in eval_metric:
                best_metric[key] = max(best_metric[key], eval_metric[key])

        if args.with_tracking:
            accelerator.log(
                {
                    "metric" if args.task_name is not None else "glue": eval_metric,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                # step=epoch,
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        logger.info(f"mnli-mm: {eval_metric}")
    
    wandb_link = accelerator.trackers[0].run.url if args.with_tracking and args.report_to == 'wandb' and accelerator.is_main_process else None
    if args.with_tracking:
        accelerator.end_training()
    
    if args.output_dir is not None and accelerator.is_main_process:
        all_results = {f"eval_{k}": v for k, v in eval_metric.items()}
        for k, v in best_metric.items():
            all_results[f"best_{k}"] = v
        all_results["wandb_link"] = wandb_link
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)

        

if __name__ == "__main__":
    main()
