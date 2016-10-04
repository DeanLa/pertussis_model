import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.integrate import odeint
from time import clock, sleep
from datetime import datetime as dt
from pprint import pprint
import pickle
from copy import copy
# import logging
# from pertussis import logger
from pertussis import *

logger.setLevel(logging.INFO)

# Initial
r_start = 1880
r_end = 1960
step = 1 / N
t_end = expand_time(r_end, start=r_start, step=step)
t_start = expand_time(r_start, start=r_start, step=step)
t_range = np.arange(t_start, t_end, 1)
vars = copy(locals())
times = []
# Data
# report_rate = np.append(nums(250, 7), nums(400, J - 7))  # Danny Cohen paper. pg3
data_years, years = cases_yearly()  # per 1e5
# data_years *= 400 / 1e5
data_months, months = cases_monthly()  # per 1e5
data_months *= 400 / 1e5
# data = cases_month_age()

# Parameters
state_0 = collect_state0()
state_0 = pack_flat(state_0)
# sys.exit("After collect state 0")
###################################################################################################################
#########################                            Run Model                   ##################################
###################################################################################################################

# Priors
o = 8
phi = 2
f_top = 25
divi = 1
f1 = 4 / divi
f2 = 0.4 / divi
f3 = 0.1 / divi
f = np.concatenate((nums(f1, sc[0]), nums(f2, sc[1]), nums(f3, sc[2])))

p = 1 / 100

# Show mean values fit
clk = clock()
RES = odeint(hetro_model, state_0, t_range,
             args=(o, phi, f, r_start))
logger.info(" " * 20 + "TIME{:10.5}".format(clock() - clk))
# # Results
# RES *= omega_ap
x = reduce_time(t_range, start=r_start, step=step)
y = unpack(RES.T, *unpack_values)
h = sum([i for i in y[:3]])
all = sum([i for i in y])
y.append(h)
y.append(all)

# PLOT
names = ["Susceptible", "Vaccinated aP", "Vaccinated wP", "Infected Is", "Infected Ia", "Recovered", "Healthy", "All"]
draw_ages = [0, 1, -4, -3, -2, -1]
fig1, ax1 = draw_model(x, y[3:], names[3:], split=0, collapse=True, ages=draw_ages)
pop = np.genfromtxt('./data/demographics/birth_rate.csv', delimiter=',', usecols=[0, 2], skip_header=1)
ax1[-1].plot(pop[:, 0], 1000 * pop[:, 1], label="Real", c='k')

# fig3, ax3 = draw_model(x, y[2:3], names[2:3], split=False, collapse=False, ages=draw_ages)

fig2, ax2 = plt.subplots()
ax2.plot(x, y[3].sum(axis=0), label='S')
ax2.plot(x, y[4].sum(axis=0), label='A')
ax2.legend()

fig3, ax3 = plt.subplots()
Ii = y[3].sum(axis=0)
ax3.plot(x, new_cases(Ii, gamma_s))
ax3.set_title("NEW SICK")

# fig, axs = plt.subplots()
# axs.plot(x, y[-1].sum(axis=0), label = "Sim")
# axs.plot(pop[:,0],1000*pop[:,1], label="Real")
# axs.legend()

plt.tight_layout()
plt.show()
