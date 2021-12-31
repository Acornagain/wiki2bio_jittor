#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-5-9 上午10:16
# @Author  : Tianyu Liu

import jittor as jt
from jittor import Module
import math


class fgateLstmUnit(Module):
    def __init__(self, hidden_size, input_size, field_size, scope_name):
        super().__init__()

        self._hidden_size = hidden_size
        self._input_size = input_size
        self._field_size = field_size
        self._scope_name = scope_name

        # self.W = jt.init.xavier_uniform((self._input_size + self._hidden_size, 4 * self._hidden_size))
        k = math.sqrt(1 / hidden_size)
        # self.W = jt.init.uniform((self._input_size + self._hidden_size, 4 * self._hidden_size), 'float32', -k, k)
        self.W = jt.init.xavier_uniform((self._input_size + self._hidden_size, 4 * self._hidden_size))
        self.b = jt.zeros([4 * self._hidden_size])
        # self.W1 = jt.init.xavier_uniform((self._field_size, 2 * self._hidden_size))
        # self.W1 = jt.init.uniform((self._field_size, 2 * self._hidden_size), 'float32', -k, k)
        self.W1 = jt.init.xavier_uniform((self._field_size, 2 * self._hidden_size))
        self.b1 = jt.zeros([2 * hidden_size])

    def execute(self, x, fd, s, finished = None):
        """
        :param x: (batch_size, input_size)
        :param fd: (batch_size, field_size)
        :param s: (batch_size, hidden_size), (batch_size, hidden_size)
        :param finished:
        :return:
        """
        h_prev, c_prev = s  # batch * hidden_size

        x = jt.concat([x, h_prev], 1)

        i, j, f, o = jt.split(jt.matmul(x, self.W) + self.b, self._hidden_size, 1)
        r, d = jt.split(jt.matmul(fd, self.W1) + self.b1, self._hidden_size, 1)

        # Final Memory cell
        c = jt.sigmoid(f+1.0) * c_prev + jt.sigmoid(i) * jt.tanh(j) + jt.sigmoid(r) * jt.tanh(d)  # batch * hidden_size
        h = jt.sigmoid(o) * jt.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            finished = jt.view(finished, (-1, 1))
            out = (1 - finished) * h
            state = ((1 - finished) * h + finished * h_prev, (1 - finished) * c + finished * c_prev)
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #          tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state