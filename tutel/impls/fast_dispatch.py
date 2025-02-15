# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import logging
import torch
from torch import Tensor

from .jit_compiler import IS_HIP_EXTENSION
from ..jit_kernels import sparse as jit_kernel
from ..jit_kernels.gating import fast_cumsum_sub_one
from .communicate import get_world_rank, simple_all_reduce
from . import losses

class GatingEncoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, reshaped_input: Tensor, *gates_):
        ctx.config = config
        if gates_:
          ctx.gates_h2 = [x.view(-1, 1).repeat(1, 2) if x.dtype == torch.float16 else x for x in gates_]
        else:
          ctx.gates_h2 = [ctx.config.ones_helper] * len(ctx.config.indices_)
        ctx.save_for_backward(reshaped_input)

        dispatched_input = torch.zeros([ctx.config.num_global_experts * ctx.config.capacity, ctx.config.model_dim], dtype=reshaped_input.dtype, device=reshaped_input.device)
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          ctx.config.func_fwd(g, i, l, reshaped_input, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
        return dispatched_input

    @staticmethod
    def backward(ctx: Any, dispatched_input: Tensor):
        dispatched_input = dispatched_input.contiguous()
        last_result = None
        reshaped_input = ctx.saved_tensors[0]
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          grad_data = torch.empty(reshaped_input.shape, dtype=dispatched_input.dtype, device=dispatched_input.device)
          ctx.config.func_bwd_data(g, i, l, grad_data, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
          last_result = grad_data if last_result is None else last_result + grad_data

        grad_gates = []
        if id(ctx.gates_h2[0]) != id(ctx.config.ones_helper):
          for i, l in zip(ctx.config.indices_, ctx.config.locations_):
            grad_gates1_s = torch.empty([ctx.config.sample_size,], dtype=dispatched_input.dtype, device=dispatched_input.device)
            ctx.config.func_bwd_gate(grad_gates1_s, i, l, reshaped_input, dispatched_input, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
            grad_gates.append(grad_gates1_s)
        return (None, last_result, *grad_gates)


class GatingDecoder(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, config: Any, expert_output: Tensor, *gates_):
        ctx.config = config
        if gates_:
          ctx.gates_h2 = [x.view(-1, 1).repeat(1, 2) if x.dtype == torch.float16 else x for x in gates_]
        else:
          ctx.gates_h2 = [ctx.config.ones_helper] * len(ctx.config.indices_)

        ctx.save_for_backward(expert_output)

        last_result = None
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          single_output = torch.empty([config.sample_size, config.model_dim], dtype=expert_output.dtype, device=expert_output.device)
          config.func_bwd_data(g, i, l, single_output, expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
          last_result = single_output if last_result is None else last_result + single_output
        return last_result

    @staticmethod
    def backward(ctx: Any, combined_output: Tensor):
        combined_output = combined_output.contiguous()
        expert_output = ctx.saved_tensors[0]
        grad_expert_output = torch.zeros(expert_output.shape, dtype=combined_output.dtype, device=combined_output.device)
        for g, i, l in zip(ctx.gates_h2, ctx.config.indices_, ctx.config.locations_):
          ctx.config.func_fwd(g, i, l, combined_output, grad_expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])

        grad_gates = []
        if id(ctx.gates_h2[0]) != id(ctx.config.ones_helper):
          for i, l in zip(ctx.config.indices_, ctx.config.locations_):
            grad_gates1_s = torch.empty([ctx.config.sample_size,], dtype=combined_output.dtype, device=combined_output.device)
            ctx.config.func_bwd_gate(grad_gates1_s, i, l, combined_output, expert_output, extra=[ctx.config.indices_[0].size(0), ctx.config.aligned_dim, ctx.config.capacity])
            grad_gates.append(grad_gates1_s)
        return (None, grad_expert_output, *grad_gates)


class TutelMoeFastDispatcher:

    kernel_pool = dict()
    ones_helper = None

    def __init__(self, num_global_experts, capacity, model_dim, dispatch_dtype):
        self.num_global_experts = int(num_global_experts)
        self.capacity = int(capacity)
        self.model_dim = int(model_dim)
        self.dtype = dispatch_dtype
        if IS_HIP_EXTENSION or dispatch_dtype != torch.float16:
            self.dtype = torch.float32
        self.original_dtype = dispatch_dtype
        self.aligned_dim = model_dim // (2 if self.dtype == torch.float16 else 1)
        self.is_cuda = None

    def update(self, indices_, locations_, gates_, capacity=None, is_postscore=True):
        self.indices_ = [x.to(torch.int32).view(-1) for x in indices_]
        self.locations_ = [x.to(torch.int32) for x in locations_]
        self.gates_ = [x.to(self.dtype) for x in gates_]
        self.is_postscore = is_postscore
        self.sample_size, self.capacity = int(self.indices_[0].size(0)), int(capacity) or self.capacity

        if self.is_cuda != indices_[0].is_cuda:
            self.is_cuda = indices_[0].is_cuda
            if self.is_cuda not in TutelMoeFastDispatcher.kernel_pool:
                self.func_fwd = jit_kernel.create_forward(self.dtype, indices_[0].is_cuda)
                self.func_bwd_data = jit_kernel.create_backward_data(self.dtype, indices_[0].is_cuda)
                self.func_bwd_gate = jit_kernel.create_backward_gate(self.dtype, indices_[0].is_cuda)
                TutelMoeFastDispatcher.kernel_pool[self.is_cuda] = self.func_fwd, self.func_bwd_data, self.func_bwd_gate
            else:
                self.func_fwd, self.func_bwd_data, self.func_bwd_gate = TutelMoeFastDispatcher.kernel_pool[self.is_cuda]

        if TutelMoeFastDispatcher.ones_helper is None or TutelMoeFastDispatcher.ones_helper.size(0) < self.sample_size:
            TutelMoeFastDispatcher.ones_helper = torch.ones([self.sample_size, 2], dtype=self.dtype, device=self.indices_[0].device)
        if TutelMoeFastDispatcher.ones_helper.is_cuda != self.indices_[0].is_cuda:
            TutelMoeFastDispatcher.ones_helper = torch.ones([TutelMoeFastDispatcher.ones_helper.size(0), 2], dtype=self.dtype, device=self.indices_[0].device)
        self.ones_helper = TutelMoeFastDispatcher.ones_helper

    def encode(self, data):
        if self.is_postscore:
            return GatingEncoder.apply(self, data.to(self.dtype)).to(self.original_dtype)
        else:
            return GatingEncoder.apply(self, data.to(self.dtype), *self.gates_).to(self.original_dtype)

    def decode(self, data):
        if self.is_postscore:
            return GatingDecoder.apply(self, data.to(self.dtype), *self.gates_).to(self.original_dtype)
        else:
            return GatingDecoder.apply(self, data.to(self.dtype)).to(self.original_dtype)

fast_dispatcher = TutelMoeFastDispatcher

def compute_sorted_location(x, importance_scores):
    sorted_x = x[importance_scores.argsort(dim=0)]
    sorted_cumsum = fast_cumsum_sub_one(sorted_x) * sorted_x
    return sorted_cumsum[importance_scores.argsort(dim=0).argsort(dim=0)]

def extract_critical(scores, top_k, loss_fn=losses.gshard_loss, capacity_factor=1.0, batch_prioritized_routing=False, normalize_gate=True, alignment=1, group=None, inequivalent_tokens=False):
    # 获取 scores 的列数，即全局专家的数量
    num_global_experts = int(scores.size(1))
    # 确保 top_k 不超过全局专家数。
    top_k, top_k_original = min(top_k, num_global_experts), top_k
    # 使用 torch.topk 函数找到每行得分最高的前 top_k 个专家的索引。
    topk_indices = torch.topk(scores, top_k, dim=1).indices
    # ruirui debug topk_indices.shape=torch.Size([8192, 2]) topk_indices=tensor([[0, 2], [2, 0],...],device='cuda:0')
    
    # 将 topk_indices 拆分成 top_k 份，并将每一份展平。
    # 结果是一个包含 top_k 个张量的列表，每个张量包含每一行的前 top_k 个专家的索引。
    indices_s = [x.view(-1) for x in topk_indices.chunk(top_k, dim=1)] # chunk方法用于将张量沿着指定维度切分为多个小张量。top_k 表示要切分的块数。dim=1 指定沿着第二维度（通常是列）进行切分。
    # ruirui debug len(indices_s)=2, len(indices_s[0])=8192, indices_s=[tensor([0, 2, 0,  ..., 3, 1, 3], device='cuda:0'), tensor([2, 0, 3,  ..., 0, 0, 1], device='cuda:0')]

    # 对每个拆分后的索引创建一个 one-hot 编码的掩码。
    masks_se = [losses._one_hot_with_dtype(x, num_classes=num_global_experts, dtype=x.dtype) for x in indices_s]
    # ruirui debug masks_se=[tensor([[1, 0, 0, 0], [0, 0, 1, 0], ...], device='cuda:0'), tensor([[0, 0, 1, 0], [1, 0, 0, 0],...], device='cuda:0')]

    # ruirui debug scores.shape=torch.Size([8192, 4]), scores=tensor([[0.3953, 0.1363, 0.2993, 0.1691], [0.2715, 0.2004, 0.3399, 0.1882], ...], device='cuda:0')
    # 计算每个样本对应的门控值，即将得分矩阵与掩码相乘后求和。
    gates_s = [(scores * x).sum(dim=1) for x in masks_se]

    # 如果提供了损失函数 loss_fn，则计算损失值。
    l_loss = loss_fn(scores, topk_indices) if loss_fn is not None else None

    if batch_prioritized_routing:
        importance_scores = -1 * scores.max(dim=1)[0]
        compute_location = lambda x: compute_sorted_location(x, importance_scores)
    else:
        compute_location = fast_cumsum_sub_one

    locations1 = compute_location(masks_se[0])

    locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

    if top_k > 1: # 如果 top_k 大于 1，则依次计算每个专家的位置，并累加到 acc_base 中
        acc_base = None
        for k in range(1, top_k):
            acc_base = torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True)
            locations2 = compute_location(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))

        # 如果需要归一化门控值，则计算所有门控值的总和，并除以这个总和。
        if normalize_gate:
            denom_s = torch.clamp(sum(gates_s), min=torch.finfo(gates_s[0].dtype).eps)
            gates_s = [x / denom_s for x in gates_s]
    else: # 如果 top_k 等于 1，则 locations2 等于 locations1。
        locations2 = locations1
    locations2 = locations2[-1] + 1

    # 将索引转换为整数类型。
    indices_s = [x.to(torch.int32) for x in indices_s]

    if inequivalent_tokens: # 如果处理不等效令牌，则通过分布式通信获得所有进程上的样本数的最大值。
        num_samples = torch.tensor(scores.size(0), device=scores.device)
        num_samples = int(simple_all_reduce(num_samples, group=group, op=torch.distributed.ReduceOp.MAX))
    else: # 否则，直接使用 scores 的行数作为样本数。
        num_samples = int(scores.size(0))

    # 计算每个专家应该处理的样本数。
    samples_per_expert = (num_samples + num_global_experts - 1) // num_global_experts

    # 如果 capacity_factor 大于 0，则根据容量因子计算容量。
    if capacity_factor > 0:
        capacity = top_k * int(capacity_factor * samples_per_expert)
    # 否则，取 locations2 的最大值，并通过分布式通信获得所有进程上的最大值。
    else:
        capacity = locations2.max()
        capacity = int(simple_all_reduce(capacity, group=group, op=torch.distributed.ReduceOp.MAX))
        # 如果 capacity_factor 小于 0，则取 capacity 和按负 capacity_factor 计算的值中的较小值。
        if capacity_factor < 0:
            capacity = min(capacity, top_k * int(-capacity_factor * samples_per_expert))

    # 如果容量不是 alignment 的倍数，则调整容量使其成为 alignment 的倍数。
    remainder = capacity % alignment
    if remainder > 0:
        capacity = capacity + alignment - remainder

    # 如果当前进程是主进程，则记录容量和实时容量因子的信息。
    if get_world_rank(group) == 0:
        logging.info(f"Capacity = {capacity}, real-time capacity-factor for top-{top_k_original} = {capacity / (top_k * samples_per_expert)}")

    # 返回一个元组，包含各种计算结果和损失值。
    return (num_global_experts, indices_s, locations_s, gates_s, capacity, locations2), l_loss

def get_dispatch_count(critial_data):
    return critial_data[-1]

def fast_encode(data, critial_data, is_postscore=True):
    assert data.is_contiguous(), "Input tensor for encode/decode should be in contiguous memory format."
    num_global_experts = critial_data[0]
    dispatcher = TutelMoeFastDispatcher(num_global_experts, 0, data.size(-1), data.dtype)
    dispatcher.update(*critial_data[1:-1], is_postscore=is_postscore)
    return dispatcher.encode(data).view(num_global_experts, -1, data.size(-1))

def fast_decode(data, critial_data, is_postscore=True):
    assert data.is_contiguous(), "Input tensor for encode/decode should be in contiguous memory format."
    num_global_experts = critial_data[0]
    dispatcher = TutelMoeFastDispatcher(num_global_experts, 0, data.size(-1), data.dtype)
    dispatcher.update(*critial_data[1:-1], is_postscore=is_postscore)
    return dispatcher.decode(data).view(-1, data.size(-1))
