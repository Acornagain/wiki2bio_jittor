#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:35
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import numpy as np
import math
import pickle


class AttentionWrapper(Module):
    def __init__(self, hidden_size, input_size, scope_name):
        super().__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._scope_name = scope_name

        # self.Wh = jt.array(np.load('/root/root/Test/att_wh.npy').astype(np.float32))
        # self.bh = jt.array(np.load('/root/root/Test/att_bh.npy').astype(np.float32))
        # self.Ws = jt.array(np.load('/root/root/Test/att_ws.npy').astype(np.float32))
        # self.bs = jt.array(np.load('/root/root/Test/att_bs.npy').astype(np.float32))
        # self.Wo = jt.array(np.load('/root/root/Test/att_wo.npy').astype(np.float32))
        # self.bo = jt.array(np.load('/root/root/Test/att_bo.npy').astype(np.float32))
        
        k = math.sqrt(1 / hidden_size)

        self.Wh = jt.init.uniform((input_size, hidden_size), 'float32', -k, k)
        self.bh = jt.init.uniform((hidden_size), 'float32', -k, k)
        self.Ws = jt.init.uniform((input_size, hidden_size), 'float32', -k, k)
        self.bs = jt.init.uniform((hidden_size), 'float32', -k, k)
        self.Wo = jt.init.uniform((2 * input_size, hidden_size), 'float32', -k, k)
        self.bo = jt.init.uniform((hidden_size), 'float32', -k, k)

        # self.Wh = jt.init.xavier_uniform((input_size, hidden_size), 'float32')
        # self.bh = jt.init.uniform((hidden_size), 'float32')
        # self.Ws = jt.init.xavier_uniform((input_size, hidden_size), 'float32')
        # self.bs = jt.init.uniform((hidden_size), 'float32')
        # self.Wo = jt.init.xavier_uniform((2 * input_size, hidden_size), 'float32')
        # self.bo = jt.init.uniform((hidden_size), 'float32')

        self.tanh = jt.nn.Tanh()

    def execute(self, x, hs, finished = None):
        hs = jt.permute(hs, [1, 0, 2])
        hs2d = jt.reshape(hs, [-1, self._input_size])
        phi_hs2d = self.tanh(jt.nn.matmul(hs2d, self.Wh) + self.bh)
        phi_hs = jt.reshape(phi_hs2d, hs.shape)
        
        gamma_h = self.tanh(jt.nn.matmul(x, self.Ws) + self.bs)
        weights = jt.sum(phi_hs * gamma_h, dim=2, keepdims=True)
        weights = jt.exp(weights - jt.max(weights, dim=0, keepdims=True))
        weights = jt.divide(weights, (1e-6 + jt.sum(weights, dim=0, keepdims=True)))
        context = jt.sum(hs * weights, dim=0)
        out = self.tanh(jt.nn.matmul(jt.concat([context, x], -1), self.Wo) + self.bo)
        if finished is not None:
            finished = jt.view(finished, [-1, 1])
            finished = jt.float32(finished)
            out = (1 - finished) * out
        return out, weights

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    att_unit = AttentionWrapper(500, 500, 'attention')

    att_unit.Wh = jt.array(np.load('/root/root/Test/dual_Wh.npy'), dtype=jt.float32)
    att_unit.bh = jt.array(np.load('/root/root/Test/dual_bh.npy'), dtype=jt.float32)

    att_unit.Ws = jt.array(np.load('/root/root/Test/dual_Ws.npy'), dtype=jt.float32)
    att_unit.bs = jt.array(np.load('/root/root/Test/dual_bs.npy'), dtype=jt.float32)

    att_unit.Wo = jt.array(np.load('/root/root/Test/dual_Wo.npy'), dtype=jt.float32)
    att_unit.bo = jt.array(np.load('/root/root/Test/dual_bo.npy'), dtype=jt.float32)

    hs = np.load('/root/root/Test/att_hs.npy')
    x = np.load('/root/root/Test/att_x.npy')
    hs = jt.array(hs, dtype=jt.float32)
    x = jt.array(x, dtype=jt.float32)
    result = att_unit(x, hs)
    print(result[0])
    print("--------------------------")
    print(result[1])