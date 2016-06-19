from .funcs import *
from .parameters import *
from .charts import *
from .cases import *
from .model import * # Requires Parameters


# import numpy as np
# from numpy import cos, pi
# from itertools import chain
# def check(x=2):
#     print("1asdas")
#     print(x)
#
#
# def beta(t, omega, phi):
#     return (1 + cos((2 * pi) * (phi / omega + t / omega)))
#
#
# def pack_flat(Y):
#     '''Takes a format of result and makes into a 1-d array. Usually before ODEint solver'''
#     x = tuple(chain(*(i if isinstance(i, tuple) \
#                           else tuple(i.flatten()) if isinstance(i, np.ndarray) \
#         else (i,) for i in Y)))
#     return np.array(x)
#
#
# def unpack(Y, *args):
#     '''Takes a 1-d array and list of and unpacks them as requested'''
#     sum_args = 0
#     for a in args:
#         if type(a) is list or type(a) is tuple:
#             sum_args += a[0] * a[1]
#         else:
#             sum_args += a
#     assert len(Y) == sum_args, "Length of Y must be equal the bins"
#     res = []
#     idx = 0
#     for i, arg in enumerate(args):
#         if type(arg) is int:
#             # print ("INT")
#             res.append(Y[idx:idx + arg])
#             idx += arg
#         elif type(arg) is list or type(arg) is tuple:
#             # print ("ELSE")
#             l = arg[0] * arg[1]
#             tmp = Y[idx:idx + l]
#             tmp = np.array(tmp).reshape(arg[0], arg[1])
#             res.append(tmp)
#             idx += l
#     return res
#
#
# def reduce_time(t, start=1948, step=1):
#     '''Takes the number of days since @param:start and return the time (usually year)'''
#     return (t - start) / step + start
#
#
# def expand_time(t, start=1948, step=1):
#     '''Takes a time (usually year) and returns the number of days since @param:start'''
#     return start + (t - start) * step
#
#
# def reduce_month(vec):
#     months = (31, 28, 31, 30,
#               31, 30, 31, 31,
#               30, 31, 30, 31)
#     l = vec.size
#     assert l % 365 == 0, "Vector must divide with 365"
#     vec = vec.reshape(int(l / 365), 365)
#     np.apply_along_axis(unpack, 1, vec, months)
#     return vec
#
#
# def reduce_year(vec):
#     '''Takes daily results from model and gives sum by years(365 days in a year)'''
#     l = vec.size
#     assert l % 365 == 0, "Vector must divide with 365, but has {} values".format(vec.shape)
#     vec = vec.reshape(l // 365, 365)
#     # np.apply_along_axis(unpack, 1, vec, months)
#     # print
#     return vec.sum(axis=1)
#
#



