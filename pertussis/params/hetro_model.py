import numpy as np
from numpy import cos, pi

# contacts = np.genfromtxt('./data/mossong.csv', delimiter=',')
AGE = 15
contacts = np.ones([AGE, AGE]) * 1 / AGE
ETH = 3
unpack_values = [AGE] * 6

def collect_state0():
    normalizer = np.ones(AGE) / AGE
    _pop = np.ones(1) / AGE
    # Compartments (State 0)
    S = 0.2 * normalizer
    Vap = 0 * normalizer
    Vwp = 0 * normalizer
    Is = 1e-4 * normalizer
    Ia = 0 * normalizer  # 1e-4
    R = _pop - S - Is - Ia

    return S, Vap, Vwp, Is, Ia, R


def collect_params(t, step, **kwargs):
    T = t - 1948

    # print (C.mean())
    C = contacts#.mean() * 1000

    # Aging
    a = np.array((1 / 6, 1 / 6, 1 / 6, 1 / 2, 6,
                  6, 7, 5, 5, 5,
                  5, 5, 10, 10))
    a = (1 / a) * (1 / step)
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
    # phi_ap = np.array([2, 2, 2, 6, 72, 72])  # month [4]
    # phi_wp = np.array([2, 2, 2, 6])  # month [4]
    # phi_ap = (12 / step) * (1 / phi_ap)
    # phi_wp = (12 / step) * (1 / phi_wp)
    omega_ap = (1 / 3) * (1 / step)  # Waning
    omega_wp = (1 / 30) * (1 / step)  # Waning

    # Probabilites
    alpha_ap = 0.2  # Chance to be symptomatic from aP
    alpha_wp = 1.0  # Chance to be symptomatic from wP
    c = 0.95  # coverage

    # Efficacies
    epsilon_ap = np.array((0.55, 0.75, 0.84, 0.98, 0.98, 0.98))  # [3]
    epsilon_wp = np.ones(4) * 0.9
    # Multiply the last value to create length of AGE
    epsilon_ap = np.concatenate((epsilon_ap, epsilon_ap[-1] * np.ones(AGE - 6)))
    epsilon_wp = np.concatenate([epsilon_wp, epsilon_wp[-1] * np.ones(AGE - 4)])

    return delta, lambda_s, lambda_a, gamma_a, \
           gamma_s, omega, mu, alpha_ap, \
           alpha_wp, c, \
           epsilon_ap, epsilon_wp, omega_ap, omega_wp, \
           a


'''Supplement:
[1] Neal Ferguson - A change in
[2] Headi Dangelis - Priming with...
[3] Meagan C F - Cost Effectiveness of Next gen...
[4] IMoH policy


Age Groups:
0. 0 - 2m
1. 2m - 4m
2. 4m - 6m
3. 6m - 1y
4. 1y - 7y
5. 7y-13y
6. 13y-20y
7. 20-25
8. 25-30
9. 30-35
10. 35-40
11. 40-45
12. 45-55
13. 55-65
14. 65 +

(1) Assuming 1 Vaccine with changing efficacies. Otherwise it's never ending if someone lost immunity on 2nd vaccine,
then what is their new efficacy after next vaccine? Assuming goes back to the normal efficacy
'''
