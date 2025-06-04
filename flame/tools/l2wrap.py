# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y, factor=1e-4):
        ctx.save_for_backward(y)
        ctx.factor = factor
        return loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        y = ctx.saved_tensors[0]
        factor = ctx.factor
        # to encourage the logits to be close to 0
        factor = factor / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy, None)  # None for the factor parameter 