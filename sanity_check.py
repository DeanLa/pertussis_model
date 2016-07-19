import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
from scipy.integrate import odeint
from copy import copy
# get_ipython().magic('matplotlib inline')

from time import clock, sleep
from datetime import datetime as dt
from pprint import pprint
from pertussis import *

report_rate = np.ones(J) * 400  # Danny Cohen paper. pg3
report_rate = np.append(np.ones(7) * 250, np.ones(4) * 1000)  # Danny Cohen paper. pg3
data_years, years = cases_yearly()  # per 1e5
data_years *= 400 / 1e5
data_months, months = cases_monthly()  # per 1e5
data_months *= 400 / 1e5

# 1998 Distribution
dist98 = get_dist_98()

# Yearly average cases per 1e5
cases = np.genfromtxt('./data/_imoh/cases.csv', skip_header=1, delimiter=',', usecols=3, filling_values=-1000)
h = np.histogram(cases, bins=np.append(a_l, 120))[0] / 16

real_cases = 1e5 * h * report_rate / (5970.6885 * 1e3 * dist98)  # per 1e5
real_cases /= 1e5  # fraction

# # Initial Parameters
r_start = 1900
r_end = 2020
step = 1 / N
t_end = expand_time(r_end, start=r_start, step=step)
t_start = expand_time(r_start, start=r_start, step=step)
t_range = np.arange(t_start, t_end, 1)
# print (t_range)
cut98 = expand_time(1998, r_start)

# Show mean values fit
state_0 = collect_state0(S0=0.2, Is0=1e-3)
state_0 = pack_flat(state_0)

f1, f2, f3 = 20, 1, 0.1
s1, s2 = 4, 3
s3 = J - s1 - s2
f = np.concatenate([nums(f1, s1), nums(f2, s2), nums(f3, s3)])
o = 4
p = 2
z = 1
vars = copy(locals())
# Run ODEs ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
clk = clock()
RES = odeint(hetro_model, state_0, t_range,
             args=(o, p, f, z, r_start))
print("ODE: ", clock() - clk)

# Prepare results
x = reduce_time(t_range, start=r_start, step=step)
y = unpack(RES.T, *unpack_values)
clk = clock()
y_new = new_cases(x, y[0], y[1], y[2], y[3], y[4], f=f, zeta=z, o=o, p=p)
print("y_new ", clock() - clk)
y_age = sum([a for a in y]).T  # By Age

h = sum([i for i in y[:3]])  # H = S + Va + Vw
all_compartments = sum([i for i in y])
y.append(h)
y.append(all_compartments)  # y = S, Va, Vw, Is, Ia, R, H, ALL
clk = clock()

print("Report: ", clock() - clk)
# print (y_new)
# print (y_new.shape)
# print (reduce_year(y_new))
# print (reduce_year(y_new).shape)
# # PLOT
figs = []
# # Healthy cases
fig2, ax2 = draw_model(x, y[0:3], ["S", "Va", "Vw"], split=None, collapse=False)
figs.append(fig2)
# Age demographics
# plt.plot(x, y_age)

# Infected
yi = (y[3] + y[4]).T.sum(axis=1)
fig3, ax3 = plt.subplots(figsize=(11, 7))
# ax3.plot(x,yi)
ax3.plot(x, y_new.sum(axis=0), 'b--', label="New Cases")
ax3.plot(x, yi, 'r--', label="Infected")
ax3.scatter(years, data_years / 12)
ax3.scatter(months, data_months, c='green')
ax3.legend()
figs.append(fig3)

# Compartments
fig, ax = draw_model(x, y[3:], ["Infected s", "Infected a", "Recovered", "Healthy", "All"], split=0, collapse=True)
plt.tight_layout()
ax[0].scatter(years, data_years / 12)
ax[0].scatter(months, data_months, c='green')
# ax[0].set_xlim(1948, 2015)
ax[0].set_ylim(0, 0.005)
figs.append(fig)

write_report(vars, x=x, y=y, figs=figs)
# fig5, ax5 = draw_model(x, [y_new], "NEW CASES")


# plt.show()
