#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:34
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import numpy as np
import math
import pickle


class LstmUnit(Module):
    def __init__(self, hidden_size, input_size, scope_name):
        super().__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._scope_name = scope_name

        k = math.sqrt(1 / hidden_size)
        self.W = jt.init.uniform((self._input_size + self._hidden_size, 4 * self._hidden_size), 'float32', -k, k)
        # self.W = jt.init.xavier_uniform((self._input_size + self._hidden_size, 4 * self._hidden_size), 'float32')
        # tmp = np.load('/root/root/Test/tf_data/enc_lstm_w.npy')
        # self.W = jt.array(tmp, dtype=jt.float32)
        # self.b = jt.init.uniform([4 * self._hidden_size], 'float32', -k, k)
        self.b = jt.init.zero([4 * self._hidden_size], 'float32')
        self.tanh = jt.nn.Tanh()
        self.sigmoid = jt.nn.Sigmoid()
        

    def execute(self, x, s, finished = None):
        h_prev, c_prev = s
        
        x = jt.concat([x, h_prev], 1)
        i, j, f, o = jt.split(jt.nn.matmul(x, self.W) + self.b, self._hidden_size, 1)
        # Final Memory cell
        c = self.sigmoid(f+1.0) * c_prev + self.sigmoid(i) * self.tanh(j)
        h = self.sigmoid(o) * self.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            finished = jt.view(finished, [-1, 1])
            finished = jt.float32(finished)
            out = (1.0 - finished) * h
            state = ((1.0 - finished) * h + finished * h_prev,
                     (1.0 - finished) * c + finished * c_prev)
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #          tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state

if __name__ == '__main__':
    jt.flags.use_cuda = 1
    l = LstmUnit(500, 400, 'lstm')
    tmp = np.load('/root/root/Test/lstm_w.npy')
    l.W = jt.array(tmp, dtype=jt.float32)
    s = np.load('/root/root/Test/lstm_hs.npy')
    x = np.load('/root/root/Test/lstm_input.npy')
    s = (s[0], s[1])
    for i in range(100):
        result = l(x, s)
        print(result[0])
        print("--------------------------")
        print(result[1][0])
        print("--------------------------")
        print(result[1][1])
        print('\n\n')
        s = result[1]