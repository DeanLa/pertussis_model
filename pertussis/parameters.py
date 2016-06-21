import numpy as np
from numpy import cos, pi
import pymc as pm

J = 11  # Age Groups
# E = 3  # Ethnicity Groups
N = 1 / 365
M = 1e-6

_O = np.ones(J)
_Z = np.zeros(J)
# C = np.ones([J, J])  # * 1 / J  # Contact Matrix
C = np.genfromtxt('./data/mossong/contact_11.csv', delimiter=',')  # Contact Matrix
# print (C)
unpack_values = [J] * 6


# Demographics
# ============
# Birth
# delta = np.ones(100) * N / 75

# print (delta)
def get_delta():
    # delta = 1.0 ** np.arange(0, 300, 1) * N / 75
    delta = np.genfromtxt('./data/demographics/birth_rate.csv', delimiter=',',
                          skip_header=1, usecols=[3]) * N

    # print ("Y")
    delta = np.append(delta, delta[-1] * np.ones(300))
    return delta


# delta = get_delta()
# mu = delta
# delta = (1 / 75) * N  # Births Yearly
# mu = delta  # * _O  # Death [] yearly

# Aging
# a_l = np.array((0,2,4,6,8,10,12,
#                 14,16,18,20,22,2*12,
#                 3*12,4*12,5*12,25*12,45*12,65*12))
# a_u = np.array((2,4,6,8,10,12,
#                 14,16,18,20,22,2*12,
#                 3*12,4*12,5*12,25*12,45*12,65*12,75*12))

a_l = np.array((0, 2 / 12, 4 / 12, 6 / 12, 1,
                7, 13, 20, 25, 45, 65))
a_u = np.array((2 / 12, 4 / 12, 6 / 12, 1,
                7, 13, 20, 25, 45, 65, 75))
# a = N * np.array((1 / 6, 1 / 6, 1 / 6, 1 / 2, 6,
#                   6, 7, 5, 5, 5,
#                   5, 5, 10, 10))
# a_u = a_u / 12
# a_l = a_l / 12
a = N / (a_u - a_l)[:-1]
# print (a, a.size, a_u.size)

# Constant Params
# ===============
# f = 1e1 * _O  # Force of infection

# Efficacy and Waning
# =====================
epsilon_ap = np.array((0.55, 0.75, 0.84, 0.98, 0.98, 0.98))  # [3]
epsilon_wp = np.ones(4) * 0.9
n_ap = epsilon_ap.size
n_wp = epsilon_ap.size
# Multiply the last value to create length of AGE
epsilon_ap = np.concatenate((epsilon_ap, epsilon_ap[-1] * np.ones(J - 6)))
epsilon_wp = np.concatenate([epsilon_wp, epsilon_wp[-1] * np.ones(J - 4)])

omega = (1 / 30) * N  # Loss of immunity [1] 3e-5 est yearly [3] 1/30 yearly
omega_ap = (1 / 3) * N  # Waning
omega_wp = (1 / 30) * N  # Waning

gamma_s = (1 / 24)  # Healing rate Symptomatic [1] 1/6 [3] 1/25
gamma_a = (1 / 8)  # Healing rate Asymptomatic [1] 16 days [3] 8

# Probabilities
alpha_ap = 0.5  # Chance to be symptomatic from aP
alpha_wp = 1  # Chance to be symptomatic from wP


def collect_state0(S0=0.2, Is0=1e-3, death=75):
    # _pop = _O / J
    # _pop = np.append(a, death - 65)# / death
    _pop = (a_u - a_l)[:-1]
    _pop = np.append(_pop, death - 65)
    _pop /= _pop.sum()
    print(_pop)
    print(_pop.sum())
    print ()
    # Compartments (State 0)
    S = S0 * _pop
    Vap = 0 * _pop
    Vwp = 0 * _pop
    Is = Is0 * _pop
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

2016_05_17
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

2016_05_24
(5) Need to check if proportion of people at age groups does not change.
(6) The efficacy is measured by which part of the V compartments has a chance to move
to I. It is different from binary moving epsilon to V and 1-epsilon to R with binary
results. GT is probably in the middle, but this model seems good and I justify that
with how there is no cocooning effect (TODO: REF)

2016_06_14
Try a method where there are 2 obsereved data - one yearly and one monthly. Later: Set weights.

2016_06_15
Assuming death rate has some square function to interpolate where no data is present.
And linear function to extrapolate. see in function _get_death_rate_
Group (from Dan: The model includes n = 19 age groups: 0–2, 2–4, 4–6, 6–12,…,22-24 months,
 and 2–3, 3–4, 4–5, 5–25, 25-45, 45–65, >65 years)

2016_06_16
Added monthly data. Some stochastics seem to explore less than others, maybe it's related to the pymc algorithm.

2016_06_19
(1) Take IMOH data and create a bar chart of proportion of sick people
(2) try to manually fit the data of cumsums.
(3) remember to multiply proportion by Danny Cohen's serology evidence
(4) find parameters so age proportions are similar to 85
'''
