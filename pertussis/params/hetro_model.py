import numpy as np
from numpy import cos, pi

# contacts = np.genfromtxt('./data/mossong/italy_phy.csv', delimiter=',')
AGE = 15
contacts = np.ones([AGE, AGE]) * 1 / AGE
ETH = 3
unpack_values = [AGE] * 6
_O = np.ones(AGE)
_Z = np.zeros(AGE)

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
    N = 1 / step

    # Aging
    a = np.array((1 / 6, 1 / 6, 1 / 6, 1 / 2, 6,
                  6, 7, 5, 5, 5,
                  5, 5, 10, 10))
    a = (1 / a) * N
    # rates
    delta = (1 / 75) * (1.0 ** T) * N  # Births Yearly
    f = 1e-3 * _O  # Force of infection - S stay at home [2]
    gamma_s = (1 / 24)  # Healing rate Symptomatic [1] 1/6 [3] 1/25
    gamma_a = (1 / 8)  # Healing rate Asymptomatic [1] 16 days [3] 8
    omega = (1 / 30) * N  # Loss of immunity [1] 3e-5 est yearly [3] 1/30 yearly
    mu = delta #* _O  # Death [] yearly

    # Vaccinations
    omega_ap = (1 / 3) * N  # Waning
    omega_wp = (1 / 30) * N  # Waning

    # Probabilites
    alpha_ap = 0.2  # Chance to be symptomatic from aP
    alpha_wp = 0.0  # Chance to be symptomatic from wP
    c = 0.95  # coverage

    # Efficacies
    epsilon_ap = np.array((0.55, 0.75, 0.84, 0.98, 0.98, 0.98))  # [3]
    epsilon_wp = np.ones(4) * 0.9
    # Multiply the last value to create length of AGE
    epsilon_ap = np.concatenate((epsilon_ap, epsilon_ap[-1] * np.ones(AGE - 6)))
    epsilon_wp = np.concatenate([epsilon_wp, epsilon_wp[-1] * np.ones(AGE - 4)])

    return delta, mu, a, \
           gamma_a, gamma_s, \
           omega, omega_ap, omega_wp, \
           alpha_ap, alpha_wp, \
           epsilon_ap, epsilon_wp, f


'''Supplement:
[1] Neal Ferguson - A change in
[2] Headi Dangelis - Priming with...
[3] Meagan C F - Cost Effectiveness of Next gen...
[4] IMoH policy
[5] Mossong Contact Matrix

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
