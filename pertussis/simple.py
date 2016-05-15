import numpy as np
from .params.simple import collect_params


def main_model(INP, t, step):
    T = (t - 1948) / step + 1948
    print(T)
    S, I, R = INP
    lambda_, gamma, omega = collect_params(T, step)
    print (lambda_,gamma,omega)
    # Susceptible 1
    dS = - lambda_ * I * S + omega * R

    dI = lambda_ * I * S - gamma * I

    dR = gamma * I - omega * R

    Y = np.array([dS, dI, dR])
    return Y
