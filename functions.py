import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def d_sigmoid(x):
    y = sigmoid(x)
    return y * (1. - y)


def d_tan(x):
    return 1. - np.tanh(x) ** 2


def lrelu(x, alpha=1e-2):
    return np.maximum(x, x * alpha)


def d_lrelu(x, alpha=1e-2):
    dx = np.ones_like(x)
    dx[x < 0] = alpha
    return dx
