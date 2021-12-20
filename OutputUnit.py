#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:36
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import numpy as np
import math
import pickle


class OutputUnit(Module):
    def __init__(self, input_size, output_size, scope_name):
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._scope_name = scope_name

        k = math.sqrt(1 / output_size)

        # self.W = jt.init.uniform((input_size, output_size), 'float32', -k, k)
        self.W = jt.init.xavier_uniform((input_size, output_size))
        # self.W = jt.randn((input_size, output_size), 'float32')
        self.b = jt.zeros([output_size], 'float32')

    def execute(self, x, finished = None):
        out = jt.nn.matmul(x, self.W) + self.b

        if finished is not None:
            # out = jt.where(finished, jt.zeros_like(out), out)
            finished = jt.view(finished, [-1, 1])
            finished = jt.float32(finished)
            out = (1.0 - finished) * out
        return out

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    o = OutputUnit(500, 20003, 'output')
    o.W = jt.array(np.load('/root/root/Test/out_w.npy'), dtype=jt.float32)
    o_input = jt.array(np.load('/root/root/Test/out_input.npy'), dtype=jt.float32)
    print(o(o_input))
