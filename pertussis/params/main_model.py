import numpy as np
from numpy import cos, pi

contacts = np.genfromtxt('./data/mossong.csv', delimiter=',')


def collect_params(t, step, **kwargs):
    # [1] Neal Ferguson - A change in
    # [2] Headi Dangelis - Priming with...
    # [3] Meagan C F - Cost Effectiveness of Next gen...
    # [4] IMoH policy

    T = t - 1948

    # local - seasonal
    _m1 = 1.49
    _m2 = 6.8
    _omega = 47.71
    _phi = 23.14
    _seasonal = _m1 * T + _m2 * T * cos((2 * pi) * (_phi / _omega + T / _omega))
    # Contacts

    # print (C.mean())
    C = contacts.mean() * 1000

    # rates
    delta = (1 / 75) * (1.0 ** T) * (1 / step)  # Births Yearly
    beta = 6.5e-5  # Force of infection - S stay at home [2]
    # beta_a = 2 * beta  # Force of infection - A go outside [] daily
    lambda_s = beta * C
    lambda_a = lambda_s
    lambda_ = lambda_s
    gamma_s = (1 / 24)  # Healing rate Symptomatic [1] 1/6 [3] 1/25
    gamma_a = (1 / 8)  # Healing rate Asymptomatic [1] 16 days [3] 8
    omega = (1 / 30) * (1 / step)  # Loss of immunity [1] 3e-5 est yearly [3] 1/30 yearly
    mu = delta  # Death [] yearly

    # Vaccinations
    phi_ap = np.array([2, 2, 2, 6, 72, 72])  # month [4]
    phi_wp = np.array([2, 2, 2, 6])  # month [4]
    phi_ap = (12 / step) * (1 / phi_ap)
    phi_wp = (12 / step) * (1 / phi_wp)
    omega_ap = (1 / 3) * (1 / step)  # Waning
    omega_wp = (1 / 30) * (1 / step)  # Waning

    # Probabilites
    alpha_ap = 0.2  # Chance to be symptomatic from aP
    alpha_wp = 1.0  # Chance to be symptomatic from wP
    c = 0.95  # coverage

    # Efficacies
    epsilon_ap = np.array((0.55, 0.75, 0.84, 0.98, 0.98, 0.98))  # [3]
    # epsilon_wp = np.array((0,0,0,0))
    # epsilon_ap = np.ones(6) #* 0.99
    epsilon_wp = np.ones(4) * 0.9

    return delta, lambda_s, lambda_a, gamma_a, \
           gamma_s, omega, mu, alpha_ap, \
           alpha_wp, c, phi_ap, phi_wp, \
           epsilon_ap, epsilon_wp, omega_ap, omega_wp


def collect_state0():
    # Compartments (State 0)
    S = 0.2  # - 1e-4
    Is = 0.0001
    Ia = 0  # 1e-4
    Vap = (0, 0, 0, 0, 0, 0)
    Vwp = (0, 0, 0, 0)
    R = 1 - S - Is - Ia
    D = 0.01 * Is

    # return S1, S2, Ia, Is, V, R, D
    return S, Vap, Vwp, Is, Ia, R


'''Supplement:
'''
