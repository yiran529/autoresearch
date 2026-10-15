# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0


import types
import torch
import numpy as np
import torch.distributed as dist

from optimizer.onebit_utils import compressed_allreduce

# append for adam mini
from typing import Optional

class OneBitAdamMini(torch.optim.Optimizer):

    def __init__(self,
                 params,
                 lr=1e-3,
                 freeze_step=100000,
                 name_dict=None,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 amsgrad=False,
                 cuda_aware=False,
                 comm_backend_name='nccl',
                 *,
                 dim: int = 2048,
                 n_heads: int = 32,
                 n_kv_heads: Optional[int] = None,
                 verbose=True,
                 ):

        if amsgrad:
            raise RuntimeError('1-bit Adam does not support the AMSGrad variant.')

        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)

        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.comm_time = 0.0
        self.step_time = 0.0
        self.ave_step = 1
        self.bk_time = 0.0

        self.adam_freeze_key = False
        self.initialize = False
        self.freeze_step = freeze_step
        self.cuda_aware = cuda_aware
        self.enable_backward_allreduce = True

        self.comm_backend_name = comm_backend_name

        assert dist.is_initialized(), "Please initialize the torch distributed backend."
        # Empty initializer. Set handle based on the comm backend as follows.
        self.size = dist.get_world_size()
        self.divider = int(self.size * 8 / np.gcd(self.size, 8))

        # Adam-mini specific
        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads

        # Setting up the names of the layers for Adam-mini
        # Embedding layer. Use AdamW updates for this block
        if name_dict:
            self.embd_names = name_dict["embd_names"]
            # Output layers. Use AdamW updates for this block
            self.output_names = name_dict["output_names"]
            # Query and Keys, will assign lrs by heads
            # for llama
            # self.wqk_names = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}
            # for Bert
            self.wqk_names = name_dict["wqk_names"]
            # MLPs
            self.mlp_names = name_dict["mlp_names"]
        else:
            # Default names for the layers, bert and llama
            self.embd_names = {"embeddings"}
            self.output_names = {"classifier"}
            self.wqk_names = {"query.weight", "key.weight"}
            self.mlp_names = {"intermediate.dense", "output.dense"}

        # Check if the model has the required layers and initialize the optim_groups
        optim_groups = []
        count_embd = count_output = count_wqk = 0

        for param_group in params:
            weight_decay = param_group.get('weight_decay', 0.0)
            for param_name, param in zip(param_group['names'], param_group['params']):
                if not param.requires_grad:
                    continue
                if verbose:
                    print('Adam-mini found the param block with name:', param_name)
                state = {}
                state["name"] = param_name
                state["params"] = param
                state["weight_decay"] = weight_decay
                if any(embd_name in param_name for embd_name in self.embd_names):
                    count_embd += 1
                if any(output_name in param_name for output_name in self.output_names):
                    count_output += 1
                if any(wqk_name in param_name for wqk_name in self.wqk_names):
                    count_wqk += 1
                    assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
                    state["head_numel"] = self.dim * self.dim // self.n_heads
                optim_groups.append(state)
        if verbose:
            print(
                f'Adam-mini found {count_embd} embedding layers, {count_output} output layers, {count_wqk} Querys and Keys.')
        if count_embd == 0 and verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, please check the name of your embedding layer and manually add them to 'self.embd_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.embd_names.add('the name of your embedding layer'). ")
        if count_output == 0 and verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No output layer found. If you are training Transformers (without weight-tying), please check the name of your output layer and manually add them to 'self.output_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.output_names.add('the name of your output layer').  Please ignore this warning if you are using weight-tying.")
        if count_wqk == 0 and verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Query or Key found. If you are training Transformers, please check the name of your Query and Key in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. You can do this by adding two additional lines of code: optimizer.wqk_names.add('the name of your Query' ); optimizer.wqk_names.add('the name of your Key'). ")
        if (count_output + count_embd + count_wqk == 0) and verbose:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. It can cause training instability on large-scale Transformers.")
        super(OneBitAdamMini, self).__init__(optim_groups, defaults)



    # add no_gard, don't sure whether works well
    @torch.no_grad()
    def step(self, closure=None, grads=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            grads_group = [None] * len(self.param_groups)
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0]) != list:
            grads_group = [grads]
        else:
            grads_group = grads

        for group, grads_this_group in zip(self.param_groups, grads_group):
            if grads_this_group is None:
                grads_this_group = [None] * len(group['params'])
            
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # length group['params'] is 1 here
            for p, grad in zip(group['params'], grads_this_group):
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('1-bit Adam does not support sparse gradients')

                state = self.state[p]

                # TODO: Adam mini loop
                is_embd_output = any(embd_name in name for embd_name in self.embd_names) or any(output_name in name for output_name in self.output_names)
                is_wqk = any(wqk_name in name for wqk_name in self.wqk_names)
                is_mlp = any(mlp_name in name for mlp_name in self.mlp_names)

                # State initialization
                if len(state) == 0:
                    if is_embd_output:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)
                    elif is_wqk:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p.data).reshape(-1, group["head_numel"])
                        state["head_per_gpu"] = state["exp_avg"].size(0)
                        state["vmean"] = torch.zeros_like(state["exp_avg"][0:state["head_per_gpu"], 0:1], memory_format=torch.preserve_format)
                    else:
                        state['step'] = 0
                        state["exp_avg"] = torch.zeros_like(p.data)
                        state["vmean"] = torch.zeros_like(torch.sum(p * p), memory_format=torch.preserve_format)
                        state["block_numel"] = torch.tensor(p.numel() * self.size, dtype=torch.float32, device=p.device)

                # EF buffers initialization
                if not self.initialize or (self.adam_freeze_key and 'worker_error' not in state.keys()):
                    state['tensor_size'] = torch.numel(p.data)
                    state['corrected_tensor_size'] = state['tensor_size']

                    if state['tensor_size'] % (self.size * self.divider) != 0:
                        state['corrected_tensor_size'] += ((self.size * self.divider) - (state['tensor_size'] %
                                                                                         (self.size * self.divider)))
                    state['server_chunk_size'] = state['corrected_tensor_size'] // self.size
                    torch.cuda.empty_cache()
                    state['worker_error'] = torch.zeros(state['corrected_tensor_size'], device=p.device)
                    state['server_error'] = torch.zeros(state['server_chunk_size'], device=p.device)
                    torch.cuda.empty_cache()
                    if dist.get_rank() == 0:
                        print("Cupy Buffers Initialized Successfully.")
                
                # Update the step
                state['step'] += 1
                beta1, beta2 = group['betas']
                exp_avg = state['exp_avg']
                
                if self.enable_backward_allreduce:
                    dist.all_reduce(grad)
                    grad.mul_(1 / self.size)

                # Full precision Adam, enabled when the freeze step is not reached or non_freeze is True
                if not self.adam_freeze_key or ('non_freeze' in group.keys() and group['non_freeze']):

                    # Comparsion between full precision and compressed
                    if self.adam_freeze_key and ('non_freeze' in group.keys() and group['non_freeze']):
                        dist.all_reduce(grad)
                        grad.mul_(1 / self.size)

                    # Full precision Adam
                    if is_embd_output:
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        del grad
                        update = exp_avg / (state['exp_avg_sq'].sqrt() + eps)
                    elif is_wqk:
                        # calculate mean squared gradient
                        grad = grad.view(state["head_per_gpu"], -1)
                        mean_sq_grad = torch.mean(grad * grad, dim=1, keepdim=True)
                        # calculate update
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)    # [head_per_gpu, head_numel]
                        state["vmean"].mul_(beta2).add_(mean_sq_grad, alpha=1 - beta2)  # [head_per_gpu, 1]
                        del grad
                        update = (exp_avg / (state["vmean"].sqrt() + eps)).view(p.size())
                    else:
                        mean_sq_grad = torch.mean(grad * grad)
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        state['vmean'].mul_(beta2).add_(mean_sq_grad, alpha=1 - beta2)
                        del grad
                        update = exp_avg / (state['vmean'].sqrt() + eps)
                # After the freeze step, we start the compressed communication
                else:
                    # Still full precision Adam for embedding and output layers
                    if is_embd_output:
                        # dist.all_reduce(grad)
                        # grad.mul_(1 / self.size)
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        # state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        del grad
                        update = exp_avg / (state['exp_avg_sq'].sqrt() + eps)
                    elif is_wqk:
                        # use grad as a temporary buffer
                        # print(f"Rank {dist.get_rank()}, Name: {name}, Step: {state['step']}")
                        # print(f"Head per GPU: {state['head_per_gpu']}")
                        grad = grad.view(state["head_per_gpu"], -1)
                        # mean_sq_grad = torch.mean(grad * grad, dim=1, keepdim=True)
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        del grad
                        # communicate exp_avg
                        if self.size > 1:
                            if dist.get_rank() == 0:
                                print(f"Step {state['step']}: Shape {state['exp_avg'].shape}", flush=True)
                            exp_avg.set_(compressed_allreduce(exp_avg, state['worker_error'], state['server_error'], dist.get_rank()))
                            # dist.all_reduce(mean_sq_grad)
                            # mean_sq_grad.mul_(1 / self.size)
                        if 'exp_avg_mask' in group:
                            if exp_avg.device != group['exp_avg_mask'].device:
                                group['exp_avg_mask'] = group['exp_avg_mask'].to(device=exp_avg.device)
                            if group['exp_avg_mask'].shape != exp_avg.shape:
                                group['exp_avg_mask'] = group['exp_avg_mask'].view(exp_avg.shape)
                            exp_avg.mul_(group['exp_avg_mask'])
                        # communicate mean squared gradient
                        # calculate update
                        # state["vmean"].mul_(beta2).add_(mean_sq_grad, alpha=1 - beta2)
                        update = (exp_avg / (state["vmean"].sqrt() + eps)).view(p.size())
                    else:
                        # use grad as a temporary buffer
                        # mean_sq_grad = torch.mean(grad * grad)
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        del grad
                        # communicate exp_avg
                        if self.size > 1:
                            if dist.get_rank() == 0:
                                print(f"Step {state['step']}: Shape {state['exp_avg'].shape}", flush=True)
                            exp_avg.set_(compressed_allreduce(exp_avg, state['worker_error'], state['server_error'], dist.get_rank()))
                            # mean_sq_grad.mul_(1 / self.size)
                        if 'exp_avg_mask' in group:
                            if exp_avg.device != group['exp_avg_mask'].device:
                                group['exp_avg_mask'] = group['exp_avg_mask'].to(device=exp_avg.device)
                            exp_avg.mul_(group['exp_avg_mask'])
                        # communicate mean squared gradient
                        # dist.all_reduce(mean_sq_grad)
                        # calculate update
                        # state['vmean'].mul_(beta2).add_(mean_sq_grad, alpha=1 - beta2)                        
                        update = exp_avg / (state['vmean'].sqrt() + eps)
                        

                # Update the weights. exp_avg and exp_avg_sq is zero when self.initialize is False
                if self.initialize:
                    if weight_decay > 0.0:
                        update += weight_decay * p.data
                        p.add_(update, alpha=-lr)
                else: # has been moved into the inner loop
                    print('Pop out errors', flush=True)
                    state.pop('worker_error')
                    state.pop('server_error')

        if not self.initialize:
            self.adam_freeze_key = False
            self.initialize = True
            print(f"Finished the initialization step at rank {dist.get_rank()}")
            return loss

        if not self.adam_freeze_key and state['step'] >= self.freeze_step:
            print('OneBitAdamMini - starting compressed communication')
            self.adam_freeze_key = True
            self.enable_backward_allreduce = False

        return loss

    def load_state_dict(self, state_dict):
        """
        Overrides load_state_dict() to add special handling when loading checkpoints
        """
        # Because at different stage exp_avg_mask may change (e.g.,
        # BERT pre-training seqlen 128 and 512 ), we don't use the exp_avg_mask
        # in checkpoints but always use the one user provided in training script.
        # (See example in DeepSpeedExamples/bing_bert/deepspeed_train.py.)
        # Thus here we keep the exp_avg_mask unchanged when loading checkpoint
        for i, group in enumerate(self.param_groups):
            if 'exp_avg_mask' in group:
                state_dict['param_groups'][i]['exp_avg_mask'] = group['exp_avg_mask']
            elif 'exp_avg_mask' not in group and 'exp_avg_mask' in state_dict['param_groups'][i]:
                state_dict['param_groups'][i].pop('exp_avg_mask')
        super().load_state_dict(state_dict)
        if 'step' not in self.state[self.param_groups[0]['params'][0]] or self.state[self.param_groups[0]['params'][0]]['step'] < self.freeze_step:
            if dist.get_rank() == 0:
                print("Checkpoint loaded and OneBitAdamMini warmup stage starts/continues.")
            if self.adam_freeze_key is True:
                self.adam_freeze_key = False
                self.enable_backward_allreduce = True
        else:
            if dist.get_rank() == 0:
                print("Checkpoint loaded and OneBitAdamMini compression stage starts/continues.")
            if self.adam_freeze_key is False:
                self.adam_freeze_key = True
                self.enable_backward_allreduce = False
        # We reset the compression errors when loading checkpoints for 3 reasons:
        # 1) The worker and server error at each GPU are distinct, so in current implementation
        # only rank 0's errors are saved in the checkpoint. Thus we have to reset the errors.
        # If we want to save them correctly we need O(num_gpu*model_size) memory in order to
        # gather all the error, which is a very large memory requirement. It's possible to save
        # them in a distributed way, but it will make the checkpoint saving/loading much more complicated.
        # 2) Even if we are able to save the compression errors correctly, you need to have the
        # exact same number of GPUs in order to load them correctly.
        # 3) We verified on BERT pre-training that occasionally resetting the compression error
        # at checkpoint loading does not affect the convergence.
        # However, please avoid frequent checkpoint loading which could break the error
        # compensation mechanism thus affect the convergence.
        for group in self.param_groups:
            for p in group['params']:
                if 'worker_error' in self.state[p]:
                    self.state[p].pop('worker_error')
                if 'server_error' in self.state[p]:
                    self.state[p].pop('server_error')
