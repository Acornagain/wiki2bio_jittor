#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-12 下午10:47
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import math

class dualAttentionWrapper(Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        super().__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._field_size = field_size
        self._scope_name = scope_name

        k = math.sqrt(1 / hidden_size)
        # self.Wh = jt.init.uniform((input_size, hidden_size), 'float32', -k, k)
        # self.bh = jt.init.uniform((hidden_size), 'float32', -k, k)

        # self.Ws = jt.init.uniform((input_size, hidden_size), 'float32', -k, k)
        # self.bs = jt.init.uniform((hidden_size), 'float32', -k, k)

        # self.Wo = jt.init.uniform((2 * input_size, hidden_size), 'float32', -k, k)
        # self.bo = jt.init.uniform((hidden_size), 'float32', -k, k)

        # self.Wf = jt.init.uniform((field_size, hidden_size), 'float32', -k, k)
        # self.bf = jt.init.uniform((hidden_size), 'float32', -k, k)

        # self.Wr = jt.init.uniform((input_size, hidden_size), 'float32', -k, k)
        # self.br = jt.init.uniform((hidden_size), 'float32', -k, k)

        self.Wh = jt.init.xavier_uniform((input_size, hidden_size))
        self.bh = jt.init.uniform((hidden_size), 'float32', -k, k)

        self.Ws = jt.init.xavier_uniform((input_size, hidden_size))
        self.bs = jt.init.uniform((hidden_size), 'float32', -k, k)

        self.Wo = jt.init.xavier_uniform((2 * input_size, hidden_size))
        self.bo = jt.init.uniform((hidden_size), 'float32', -k ,k)

        self.Wf = jt.init.xavier_uniform((field_size, hidden_size))
        self.bf = jt.init.uniform((hidden_size), 'float32', -k, k)

        self.Wr = jt.init.xavier_uniform((input_size, hidden_size))
        self.br = jt.init.uniform((hidden_size), 'float32', -k, k)

    def execute(self, x, hs, fds, coverage = None, finished = None, alpha = None, beta = None, gamma = None):
        '''
            :param x (batch_size, hidden_size)
            :param hs (batch_size, max_len, hidden_size)
            :param fds (batch_size, max_len, field_size)
        '''
        hs = jt.permute(hs, [1, 0, 2])  # input_len * batch_size * hidden_size
        fds = jt.permute(fds, [1, 0, 2]) # input_len * batch_size * field_size

        hs2d = jt.reshape(hs, [-1, self._input_size])
        phi_hs2d = jt.tanh(jt.matmul(hs2d, self.Wh) + self.bh)
        phi_hs = jt.reshape(phi_hs2d, hs.shape)
        # print("phi_hs\n", phi_hs)

        fds2d = jt.reshape(fds, [-1, self._field_size])
        phi_fds2d = jt.tanh(jt.matmul(fds2d, self.Wf) + self.bf)
        phi_fds = jt.reshape(phi_fds2d, hs.shape) 
        # print("phi_fd\n", phi_fds)

        gamma_h = jt.tanh(jt.matmul(x, self.Ws) + self.bs)  # batch * hidden_size
        alpha_h = jt.tanh(jt.matmul(x, self.Wr) + self.br)
        # print("gamma_h\n", gamma_h)
        # print("alpha_h\n", alpha_h)

        fd_weights = jt.sum(phi_fds * alpha_h, dim=2, keepdims=True)
        fd_weights = jt.exp(fd_weights - jt.max(fd_weights, dim=0, keepdims=True))
        fd_weights = jt.divide(fd_weights, (1e-6 + jt.sum(fd_weights, dim=0, keepdims=True)))
        # if beta is not None:
        #     beta.append(fd_weights.tolist())
        # print("fd_weights\n" ,fd_weights)

        weights = jt.sum(phi_hs * gamma_h, dim=2, keepdims=True)  # input_len * batch
        weights = jt.exp(weights - jt.max(weights, dim=0, keepdims=True))
        weights = jt.divide(weights, (1e-6 + jt.sum(weights, dim=0, keepdims=True)))
        # if alpha is not None:
        #     alpha.append(weights.tolist())
        # print("weights\n", weights)
        weights = jt.divide(weights * fd_weights, (1e-6 + jt.sum(weights * fd_weights, dim=0, keepdims=True)))
        # if gamma is not None:
        #     gamma.append(weights.tolist())
        context = jt.sum(hs * weights, dim=0)  # batch * hidden_size

        out = jt.tanh(jt.matmul(jt.concat([context, x], -1), self.Wo) + self.bo)

        if finished is not None:
            finished = jt.view(finished, [-1, 1])
            finished = jt.float32(finished)
            out = (1 - finished) * out
        return out, weights

if __name__ == '__main__':
    import numpy as np

    jt.flags.use_cuda = 1
    att_unit = dualAttentionWrapper(500, 500, 50, 'dual_attention')

    att_unit.Wh = jt.array(np.load('/root/root/Test/dual_Wh.npy'), dtype=jt.float32)
    att_unit.bh = jt.array(np.load('/root/root/Test/dual_bh.npy'), dtype=jt.float32)

    att_unit.Ws = jt.array(np.load('/root/root/Test/dual_Ws.npy'), dtype=jt.float32)
    att_unit.bs = jt.array(np.load('/root/root/Test/dual_bs.npy'), dtype=jt.float32)

    att_unit.Wf = jt.array(np.load('/root/root/Test/dual_Wf.npy'), dtype=jt.float32)
    att_unit.bf = jt.array(np.load('/root/root/Test/dual_bf.npy'), dtype=jt.float32)

    att_unit.Wo = jt.array(np.load('/root/root/Test/dual_Wo.npy'), dtype=jt.float32)
    att_unit.bo = jt.array(np.load('/root/root/Test/dual_bo.npy'), dtype=jt.float32)

    att_unit.Wr = jt.array(np.load('/root/root/Test/dual_Wr.npy'), dtype=jt.float32)
    att_unit.br = jt.array(np.load('/root/root/Test/dual_br.npy'), dtype=jt.float32)

    
    x = jt.array(np.load('/root/root/Test/dual_att_x.npy'), dtype=jt.float32)
    hs = jt.array(np.load('/root/root/Test/dual_att_hs.npy'), dtype=jt.float32)
    fds = jt.array(np.load('/root/root/Test/dual_att_fds.npy'), dtype=jt.float32)
    result = att_unit(x, hs, fds)
    print(result[0])
    print("--------------------------")
    print(result[1])