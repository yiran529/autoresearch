# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import types
import torch
import numpy as np
import torch.distributed as dist

from optimizer.onebit_utils import compressed_allreduce

class OneBitAdam(torch.optim.Optimizer):
    """Implements the 1-bit Adam algorithm. Currently GPU-only.
    For usage example please see https://www.deepspeed.ai/tutorials/onebit-adam/
    For technical details please read https://arxiv.org/abs/2102.02888

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        freeze_step (int, optional): Number of steps for warmup (uncompressed)
            stage before we start using compressed communication. (default 100000)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in 1-bit Adam!
        eps_inside_sqrt (boolean, optional): in the 'update parameters' step,
            adds eps to the bias-corrected second moment estimate before
            evaluating square root instead of adding it to the square root of
            second moment estimate as in the original paper. (default: False)
        cuda_aware (boolean, required): Set True if the underlying MPI implementation
            supports CUDA-Aware communication. (default: False)
        comm_backend_name (string, optional): Set to 'mpi' if needed. (default: 'nccl')
    .. _Adam\\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 freeze_step=100000,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 eps_inside_sqrt=False,
                 weight_decay=0.,
                 max_grad_norm=0.,
                 amsgrad=False,
                 cuda_aware=False,
                 comm_backend_name='nccl'):

        if amsgrad:
            raise RuntimeError('1-bit Adam does not support the AMSGrad variant.')

        defaults = dict(lr=lr,
                        bias_correction=bias_correction,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm)

        super(OneBitAdam, self).__init__(params, defaults)
        self.eps_mode = 0 if eps_inside_sqrt else 1
        self.comm_time = 0.0
        self.step_time = 0.0
        self.ave_step = 1
        self.bk_time = 0.0
        self.mamba=0

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

        gather_time = 0
        allgather_time = 0
        all_time = 0

        if self.adam_freeze_key is False:
            v_diff_buffer = 0.0

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

            bias_correction = 1 if group['bias_correction'] else 0
            for p, grad in zip(group['params'], grads_this_group):
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('1-bit Adam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

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
                    self.adam_freeze_key = True
                    if not self.initialize and dist.get_rank() == 0:
                        print("Cupy Buffers Initialized Successfully.")
                
                    # if dist.get_rank() == 0 and len(state) > 0:
                    # print(f"Step {state['step']}: Adam freeze key {self.adam_freeze_key}", flush=True)
                
                # Update the step
                state['step'] += 1
                beta1, beta2 = group['betas']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                if self.adam_freeze_key is False:
                    # if dist.get_rank()==0:
                    #     tem=1
                    #     for i in range(len(grad.shape)):
                    #         tem*=grad.shape[i]
                    #     self.mamba+=tem*32*8
                        # print(state['step'],grad.shape,grad.dtype)
                    dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=False)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    grad = None
                    if self.initialize:
                        update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                else:
                    if 'non_freeze' in group.keys() and group['non_freeze'] is True:
                        dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=False)
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                        grad = None
                    else:
                        if self.initialize is True:
                            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                        grad = None

                        if self.size > 1:
                            # if dist.get_rank() == 0:
                            #     print(f"Step {state['step']}: Shape {exp_avg.shape}", flush=True)
                            # TODO compressed allreduce
                            
                            exp_avg.set_(compressed_allreduce(exp_avg, state['worker_error'], state['server_error'], dist.get_rank()))
                            # exp_avg.set_(compressed_allreduce_simulate(exp_avg, state['worker_error'], state['server_error'], dist.get_rank()))
                            # tem=1
                            # for i in range(len(exp_avg.shape)):
                            #     tem*=exp_avg.shape[i]
                            # self.mamba+=(tem*1+32*int(torch.linalg.norm(exp_avg) / np.sqrt(exp_avg.numel())))*8
                        # Because 1-bit compression cannot represent exact zero, it is required to
                        # provide a momentum mask for those params that have constant exact zeros in their
                        # momentums, otherwise the compression error would keep accumulating.
                        # For example, for BERT pre-training seq 128, bert.embeddings.position_embeddings.weight
                        # always have exact zeros in its momentum for row 129 to 512, because it only
                        # learns up to seq length 128 while the model supports up to 512 seq length.
                        # (See example in DeepSpeedExamples/bing_bert/deepspeed_train.py.)
                        if 'exp_avg_mask' in group:
                            if exp_avg.device != group['exp_avg_mask'].device:
                                group['exp_avg_mask'] = group['exp_avg_mask'].to(device=exp_avg.device)
                            exp_avg.mul_(group['exp_avg_mask'])

                    if self.initialize:
                        update = exp_avg / (exp_avg_sq.sqrt() + group['eps'])

                if self.initialize:
                    if group['weight_decay'] > 0.0:
                        update += group['weight_decay'] * p.data
                    with torch.no_grad():
                        p.add_(update, alpha=-group['lr'])

            # if not self.initialize:
            #     print('Pop out errors', flush=True)
            #     state.pop('worker_error')
            #     state.pop('server_error')

        # import json
        # import os

        # file_path = '/root/workspace/cu29hik7v38s73eo9r50/jwc/Last/CE-Adam/data/test.json'

        # if dist.get_rank()==0:
        #     # 检查文件是否存在，如果不存在则创建并初始化为一个空列表
        #     if not os.path.exists(file_path):
        #         with open(file_path, 'w', encoding='utf-8') as f:
        #             json.dump([], f)  # 初始化文件为一个空列表

        #     # 打开并读取已有的JSON文件
        #     with open(file_path, 'r', encoding='utf-8') as f:
        #         data = json.load(f)  # 加载已有的数据，应该是一个列表

        #     # 追加新数字
        #     # data.append(self.mamba)

        #     # 将更新后的列表写回到JSON文件
        #     with open(file_path, 'w', encoding='utf-8') as f:
        #         json.dump(data, f, ensure_ascii=False, indent=4)  # 使用indent使输出更易读

        if not self.initialize:
            self.adam_freeze_key = False
            self.initialize = True
            print(f"Finished the initialization step at rank {dist.get_rank()}")
            return loss

        if not self.adam_freeze_key and state['step'] >= self.freeze_step:
            print('OneBitAdam - starting compressed communication')
            self.adam_freeze_key = True
            self.enable_backward_allreduce = False

        # if dist.get_rank()==0:
        #     print("mamba",self.mamba)
        
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
                print("Checkpoint loaded and OneBitAdam warmup stage starts/continues.")
            if self.adam_freeze_key is True:
                self.adam_freeze_key = False
                self.enable_backward_allreduce = True
        else:
            if dist.get_rank() == 0:
                print("Checkpoint loaded and OneBitAdam compression stage starts/continues.")
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
