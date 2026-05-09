"""LSTM Cell 手写.
PREFERRED: numpy (面试要求你写出四个门公式)

门公式 (x: (d_in,), h_prev: (d_hidden,)):
  combined = concat([x, h_prev])
  f = sigmoid(Wf @ combined + bf)    遗忘门
  i = sigmoid(Wi @ combined + bi)    输入门
  g = tanh   (Wg @ combined + bg)    候选记忆
  o = sigmoid(Wo @ combined + bo)    输出门
  c = f * c_prev + i * g
  h = o * tanh(c)
"""
import numpy as np


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


class LSTMCell:
    def __init__(self, d_in, d_hidden, seed=0):
        rng = np.random.default_rng(seed)
        D = d_in + d_hidden
        self.Wf = rng.normal(0, 0.1, (d_hidden, D)); self.bf = np.zeros(d_hidden)
        self.Wi = rng.normal(0, 0.1, (d_hidden, D)); self.bi = np.zeros(d_hidden)
        self.Wg = rng.normal(0, 0.1, (d_hidden, D)); self.bg = np.zeros(d_hidden)
        self.Wo = rng.normal(0, 0.1, (d_hidden, D)); self.bo = np.zeros(d_hidden)
        self.d_hidden = d_hidden

    def step(self, x, h_prev, c_prev):
        # TODO: 按上面公式实现, 返回 h, c
        raise NotImplementedError

    def forward(self, xs):
        """xs: (T, d_in) -> hs: (T, d_hidden)"""
        # TODO
        raise NotImplementedError
