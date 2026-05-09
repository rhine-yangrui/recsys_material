"""RNN Cell (vanilla) 手写.
PREFERRED: numpy (面试常考公式);  torch 版用来对照

递推: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
     y_t = W_hy @ h_t + b_y
"""
import numpy as np


class RNNCell:
    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.Wxh = rng.normal(0, 0.1, (d_hidden, d_in))
        self.Whh = rng.normal(0, 0.1, (d_hidden, d_hidden))
        self.Why = rng.normal(0, 0.1, (d_out, d_hidden))
        self.bh = np.zeros(d_hidden); self.by = np.zeros(d_out)

    def forward(self, xs, h0=None):
        """
        xs: (T, d_in)
        return: ys (T, d_out), hs (T, d_hidden)
        """
        # TODO: for t in range(T): h = tanh(Wxh@x + Whh@h_prev + bh); y = Why@h + by
        raise NotImplementedError
