import numpy as np
from numpy import cos, pi
import pymc as pm

J = 15  # Age Groups
# E = 3  # Ethnicity Groups
N = 1 / 365
M = 1e-6

_O = np.ones(J)
_Z = np.zeros(J)
C = np.ones([J, J]) * 1 / J  # Contact Matrix
# C = np.genfromtxt('./data/mossong/italy_phy.csv', delimiter=',') #Contact Matrix
unpack_values = [J] * 6
# Demographics
# ============
# Birth
# delta = np.ones(100) * N / 75
delta = 1.0 ** np.arange(0,100,1) * N / 75
mu = delta
# delta = (1 / 75) * N  # Births Yearly
# mu = delta  # * _O  # Death [] yearly
# Aging
a = N * np.array((1 / 6, 1 / 6, 1 / 6, 1 / 2, 6,
                  6, 7, 5, 5, 5,
                  5, 5, 10, 10))
# Constant Params
# ===============
# f = 1e1 * _O  # Force of infection

# Efficacy and Waning
# =====================
epsilon_ap = np.array((0.55, 0.75, 0.84, 0.98, 0.98, 0.98))  # [3]
epsilon_wp = np.ones(4) * 0.9
# Multiply the last value to create length of AGE
epsilon_ap = np.concatenate((epsilon_ap, epsilon_ap[-1] * np.ones(J - 6)))
epsilon_wp = np.concatenate([epsilon_wp, epsilon_wp[-1] * np.ones(J - 4)])

omega = (1 / 30) * N  # Loss of immunity [1] 3e-5 est yearly [3] 1/30 yearly
omega_ap = (1 / 3) * N  # Waning
omega_wp = (1 / 30) * N  # Waning

gamma_s = (1 / 24)  # Healing rate Symptomatic [1] 1/6 [3] 1/25
gamma_a = (1 / 8)  # Healing rate Asymptomatic [1] 16 days [3] 8

# Probabilities
alpha_ap = 0.2  # Chance to be symptomatic from aP
alpha_wp = 1  # Chance to be symptomatic from wP



def collect_state0():
    _pop = _O / J
    # Compartments (State 0)
    S = 0.1 * _pop
    Vap = 0 * _pop
    Vwp = 0 * _pop
    Is = 1e-3 * _pop
    Ia = 0 * _pop  # 1e-4
    R = _pop - S - Is - Ia

    return S, Vap, Vwp, Is, Ia, R


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
then what is their new efficacy after next vaccine? Assuming goes back to the normal efficacy.
(2) There is a need to normalize I accordingly - If the population grows larger, an individual still meets the same number
of people per day. normalization needs to happen according to share in population.
If AGE = S + V + I + R then I_t is I*AGE(t)/AGE(0). that is only normalized in the CONTACTS phase, and not for the
"real" I.
(3) there will be 3-4 f groups (susceptability levels) and f is a vector of the shape:
(f1,f1,...,f1,f2,...f2,f3,...f3)
(4) Real data is observed sometimes once a year and sometimes once a month. There is a need to address that.
if once a year error is E^2 then once a month it's approx 12*(E/12)^2 = E^2/12. The model will favor the yearly
observations in order to minimize the error.
TODO: Needs to be addressed.
'''
