import numpy as np
from numpy import cos, pi
from scipy.stats import expon
import pymc as pm
import sys
from pertussis import medlock

# Demographics
# =========================================================
# Age Groups
N = 1 / 365  # Step
_ages = np.hstack((np.arange(0, 1, 2 / 12),  # First year: Every 2 months
                   np.arange(1, 14, 1),  # Every year until age of 13
                   15, 18, 21,  # High Scholl, Grad, Army
                   np.arange(25, 66, 10),  # 10 Year gaps
                   100))  # Death
a_l = _ages[:-1]  # Lower limits
a_u = _ages[1:]  # Upper limis
a = N / (a_u - a_l)[:-1]

# Constants

C = np.genfromtxt('./data/mossong/medlock_avg_sym.csv', delimiter=',')  # Contact Matrix
C = medlock(C, _ages)

# Birth and Death
# delta = np.ones(1100) * N / 75
delta = np.genfromtxt('./data/demographics/birth_rate.csv', delimiter=',',
                      skip_header=1, usecols=[3]) * N
delta = np.pad(delta, (0, 1000), 'edge')
# print(delta)
mu = delta  # * _O  # Death [] yearly
death = _ages[-1]

# Constants
# =========================================================
J = _ages.size  # Age Groups
M = 1e-6  # Small M
unpack_values = [J] * 6
_O = np.ones(J)
_Z = np.zeros(J)

# Scenarios
# =========================================================

# Scenarios for alpha_ap = | 0.15 | 0.5 | 0.75 |
alpha_ap = 0.5  # Chance to be symptomatic from aP
alpha_wp = 1  # Chance to be symptomatic from wP

# Waning
# Scenarios for omega_ap = | 4 y | 30 y |
omega_ap = (1 / 30) * N  # Waning [6] as low as 4-12 years
omega = (1 / 30) * N  # Loss of immunity [1] 3e-5 est yearly [3] 1/30 yearly [6] as low as 7-20 years
omega_wp = omega

# Vaccines
# =========================================================
# Policy
_vax_ages = [2 / 12, 4 / 12, 6 / 12, 1, 7, 13]
assert all(np.in1d(_vax_ages, _ages)), "Vaccine should happen on listed Age Group:\n {}".format(_ages)
vax = np.in1d(_ages, _vax_ages, ).astype(int)
vax

# Efficacy
# epsilon_ap = np.ones(6) * 1
# epsilon_wp = np.ones(4) * 0.9
epsilon_ap = np.array((0.55, 0.75, 0.84, 0.98, 0.98, 0.98))  # [3]
epsilon_wp = [0.9]
n_ap = len(epsilon_ap)
n_wp = len(epsilon_wp)
# Multiply the last value to create length of AGE
epsilon_ap = np.pad(epsilon_ap, (0, J - n_ap), 'edge')
epsilon_wp = np.pad(epsilon_wp, (0, J - n_wp), 'edge')

# Recovery
# =========================================================
gamma_s = (1 / 25)  # Healing rate Symptomatic [1] 1/6 [3] 1/25
gamma_a = (1 / 8)  # Healing rate Asymptomatic [1] 16 days [3] 8
recovered_ratio = expon(scale = 1 / gamma_s).cdf(1) # Ratio of people recovered today

def collect_state0(S0=0.2, Is0=1e-3, death=death):
    # _pop = _O / J
    # _pop = np.append(a, death - 65)# / death
    _pop = (a_u - a_l)[:-1]
    _pop = np.append(_pop, death - 65)
    _pop /= _pop.sum()
    # print(_pop)
    # print(_pop.sum())
    # print ()
    # Compartments (State 0)
    S = S0 * _pop
    Vap = 0 * _pop
    Vwp = 0 * _pop
    Is = Is0 * _pop
    Ia = 0 * _pop  # 1e-4
    R = _pop - S - Is - Ia

    return S, Vap, Vwp, Is, Ia, R


sys.exit(-1)

'''Supplement:
[1] Neal Ferguson - A change in
[2] Headi Dangelis - Priming with...
[3] Meagan C F - Cost Effectiveness of Next gen...
[4] IMoH policy
[5] Transmission of Bordetella pertussis to Young Infants
[6] Duration_of_Immunity_Against_Pertussis_After.11
[7] Acellular pertussis vaccines protect against disease but PNAS SI
[8] Pertussis_Sources_of_Infection_and_Routes_of.4
# [] Mossong Contact Matrix

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
