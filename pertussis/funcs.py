import numpy as np
from numpy import cos, pi
from itertools import chain
import pickle


def check(x=2):
    print("T")
    print(x)


def beta(t, omega, phi, rho=0):
    if t >= 1900:
        return rho + cos((2 * pi * t / omega) + phi)
    else:
        return 0


def pack_flat(Y):
    """Takes a format of result and makes into a 1-d array. Usually before ODEint solver"""
    x = tuple(chain(*(i if isinstance(i, tuple) \
                          else tuple(i.flatten()) if isinstance(i, np.ndarray) \
        else (i,) for i in Y)))
    return np.array(x)


def unpack(Y, *args):
    """Takes a 1-d array and list of and unpacks them as requested"""
    sum_args = 0
    for a in args:
        if type(a) is list or type(a) is tuple:
            sum_args += a[0] * a[1]
        else:
            sum_args += a
    assert len(Y) == sum_args, "Length of Y must be equal the bins {} != {}".format(len(Y), sum_args)
    res = []
    idx = 0
    for i, arg in enumerate(args):
        if type(arg) is int:
            # print ("INT")
            res.append(Y[idx:idx + arg])
            idx += arg
        elif type(arg) is list or type(arg) is tuple:
            # print ("ELSE")
            l = arg[0] * arg[1]
            tmp = Y[idx:idx + l]
            tmp = np.array(tmp).reshape(arg[0], arg[1])
            res.append(tmp)
            idx += l
    return res


def reduce_time(days, start, step):
    """Takes the number of days since @param:start and return the time (usually year)"""
    return days / step + start


def expand_time(t, start, step):
    """Takes a time (usually year) and returns the number of days since @param:start"""
    return (t - start) * step


def reduce_month(vec):
    months = (31, 28, 31, 30,
              31, 30, 31, 31,
              30, 31, 30, 31)
    if vec.ndim == 1:
        l = vec.size
        assert l % 365 == 0, "Vector must divide with 365. Current modulo: {}".format(l % 365)
        months = np.tile(months, l // 365)
        months = np.cumsum(months)
        res = np.split(vec, months[:-1])
        res = np.array([a.sum() for a in res])
        return res
    if vec.ndim == 2:
        return np.apply_along_axis(reduce_month, 1, vec)


def reduce_year(vec):
    """Takes daily results from model and gives sum by years(365 days in a year)"""
    if vec.ndim == 1:
        l = vec.size
        assert l % 365 == 0, "Vector must divide with 365, but has {} values".format(vec.shape)
        years = np.cumsum(np.tile(365, l // 365))
        res = np.split(vec, years[:-1])
        res = np.array([a.sum() for a in res])
        return res
    if vec.ndim == 2:
        return np.apply_along_axis(reduce_year, 1, vec)


def nums(scalar, amount):
    return np.ones(amount) * scalar


def mse(x, y):
    res = (x - y) ** 2
    return res.mean()


def medlock(C, vec):
    """Takes a medlock symmetric Matrix with an ages vector, and stratifies the matrix according to the ages"""
    v = [np.arange(a, b) for a, b in zip(vec[:-1], vec[1:])]
    E = np.zeros([len(v)] * 2)
    for i in range(len(v)):
        for j in range(len(v)):
            val = 0
            cnt = 0
            for vi in v[i]:
                for vj in v[j]:
                    cnt += 1
                    val += C[min(C.shape[0] - 1, int(vi) // 5), min(C.shape[1] - 1, int(vj) // 5)]
            E[i, j] = val / cnt
    return E


def age_correction(start_year, T, age_vec):
    """Takes a start year, current year T and age vector of the upper bound of the age group (a_u)
     returns a vector of 0 and 1 to avoid transition to unwanted age groups """
    vec = (T - start_year) > age_vec
    return vec.astype(int)[:-1]


def new_cases(vec, healing):
    ret = vec[1:] - (1 - healing) * vec[:-1]
    ret = np.hstack((0, ret))
    return ret


def binom_likelihood(model, data, p):
    from scipy.special import binom
    assert all((model >= data).reshape(-1, 1)), "model<data somewhere"
    choose = binom(model, data)
    ll = np.log(choose) + data * np.log(p) + (model - data) * np.log(1 - p)
    return ll.sum()


def log_liklihood(model, data, sigma=1):
    diff = (model - data) ** 2
    LL = -diff.sum() / (2 * sigma ** 2)
    # print (LL)
    return LL


def log_ratio(model_star, model_current, data, sigma=1):
    """Compute log ratio between two models
    to compute r add between those
    https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf"""
    diff_star = log_liklihood(model_star, data, sigma)
    diff_current = log_liklihood(model_current, data, sigma)
    # diff_star = (model_star - data) ** 2
    # diff_current = (model_current - data) ** 2
    # log_ratio = - (diff_star.sum() - diff_current.sum()) / (2 * sigma ** 2)
    return diff_star - diff_current


def make_chains(mcmcs):
    chains = [mcmc['chain'] for mcmc in mcmcs]
    return chains


def gelman_rubin(chains):
    """http://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/"""
    M = len(chains)
    assert len(chains) >= 2, "test for less than 2 is redundunt"
    params = chains[0].shape[1]  # Asumming chains have equal params

    N = len(chains[0][:, 0])  # Asumming chains have equal length
    B = np.zeros(params)  # Init for values of params
    W = np.zeros(params)  # Init for values of parmas
    for i in range(params):
        means = [np.mean(chain) for chain in chains]
        variances = [np.var(chain) for chain in chains]
        B[i] = np.var(means)
        W[i] = np.mean(variances)
    V = ((N - 1) / N) * W + ((M + 1) / (M * N)) * B
    return V





def save_mcmc(obj, path='./'):
    name = obj['name']
    path = path+name+'.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_mcmc(path='./mcmc.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


