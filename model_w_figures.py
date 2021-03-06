import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
from scipy.integrate import odeint
# %matplotlib inline

from time import clock, sleep
from datetime import datetime as dt
from pprint import pprint

from imp import reload
from pertussis import *


r_start = 1948
r_end = 2050
step = 1 / N
t_end = expand_time(r_end, start=r_start, step=step)
t_start = expand_time(r_start, start = r_start, step=step)

t_range = np.arange(t_start, t_end, 1)

# report_rate = np.ones(J) * 400 # Danny Cohen paper. pg3
report_rate = np.append(np.ones(7) * 250, np.ones(4) * 1000) # Danny Cohen paper. pg3

data_years, years = cases_yearly()  # per 1e5
data_years *= 400 / 1e5

data_months, months = cases_monthly()  # per 1e5
data_months *= 400 / 1e5


# 1998 Distribution
dist98 = get_dist_98()

# Yearly average cases per 1e4
cases = np.genfromtxt('./data/_imoh/cases.csv', skip_header=1, delimiter=',',usecols=3, filling_values=-1000)
h = np.histogram(cases, bins=np.append(a_l,120))[0] / 16

real_cases = 1e5 * h * report_rate / (5970.6885 * 1e3 * dist98) # per 1e5
real_cases /= 1e5 # fraction
real_cases

# Manual Parameters

death=100
# d = 1.0 ** np.arange(0, 300, 1) * N / death
d = get_delta()
cut98 = expand_time(1998, r_start)

# Show mean values fit
state_0 = collect_state0(S0=0.8, Is0=1e-3, death=100)
state_0 = pack_flat(state_0)

_o = lambda f,s: np.ones(s)*f

f1,f2,f3 = 15.1,1.5,0.1
s1,s2 = 4,3
s3 = J-s1-s2

f = np.concatenate([_o(f1,s1),_o(f2,s2),_o(f3,s3)])
print (f)
# Run ODEs
RES = odeint(hetro_model, state_0, t_range,
             args=(4, 2, f, 1, d ))

# Prepare results
x = reduce_time(t_range, start=1948, step=step)
y = unpack(RES.T, *unpack_values)

# Sum by Age
y_age = sum([a for a in y]).T
# print (y_age)
model98 = y_age[cut98]

h = sum([i for i in y[:3]])
all_compartments = sum([i for i in y])
y.append(h)
y.append(all_compartments)

# Prepare sick people
sick = np.apply_along_axis(reduce_year, 1, y[3])[:,1998 - r_start:2014 - r_start]
model_cases = sick.mean(axis=1)

fig, ax = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=True)
plt.tight_layout()
ax[0].scatter(years, data_years)
ax[0].scatter(months, data_months, c='green')

ax[0].set_xlim(1948, 2015)
ax[0].set_ylim(0, 0.05)
plt.show()