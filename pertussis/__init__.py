import numpy as np
from numpy import cos, pi
from itertools import chain

J = 15
T = 3
def beta(t, m, omega, phi):
    return m * (1 + cos((2 * pi) * (phi / omega + t / omega)))


def pack(Y):
    x = tuple(chain(*(i if isinstance(i, tuple) \
                          else tuple(i) if isinstance(i, np.ndarray) \
        else (i,) for i in Y)))
    return np.array(x)


def unpack(Y, *args):
    assert len(Y) == (sum(args)), "Length of Y must be equal the bins"
    res = []
    idx = 0
    for i, arg in enumerate(args):
        if arg == 1:
            res.append(Y[idx])
        else:
            res.append(Y[idx:idx + arg])
        idx += arg
    return res


def reduce_time(t, start=1948, step=1):
    return (t - start) / step + start


def expand_time(t, start=1948, step=1):
    return start + (t - start) * step


def reduce_month(vec):
    months = (31, 28, 31, 30,
              31, 30, 31, 31,
              30, 31, 30, 31)
    l = vec.size
    assert l % 365 == 0, "Vector must divide with 365"
    vec = vec.reshape(int(l / 365), 365)
    np.apply_along_axis(unpack, 1, vec, months)
    return vec


def reduce_year(vec):
    l = vec.size
    assert l % 365 == 0, "Vector must divide with 365, but has {} values".format(vec.shape)
    vec = vec.reshape(l // 365, 365)
    # np.apply_along_axis(unpack, 1, vec, months)
    # print
    return vec.sum(axis=1)


# from .params import *
from .model import *
