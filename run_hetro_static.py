import numpy as np
import matplotlib.pyplot as plt
# import pymc as pm
from scipy.integrate import odeint
from time import clock, sleep
from datetime import datetime as dt
from pprint import pprint
import pickle
from copy import copy
# import logging
# from pertussis import logger
from pertussis import *

# logger.setLevel(logging.DEBUG)

# Initial
r_start = 1955
r_end = 2015
step = 1 / N
t_end = expand_time(r_end, start=r_start, step=step)
t_start = expand_time(r_start, start=r_start, step=step)
t_range = np.arange(t_start, t_end, 1)
vars = copy(locals())
times = []
# Data
# report_rate = np.append(nums(250, 7), nums(400, J - 7))  # Danny Cohen paper. pg3
# data_years, years = cases_yearly()  # per 1e5
# data_years *= 400 / 1e5
data, months = cases_monthly()  # per 1e5
logger.debug(data.shape)
logger.debug(months.shape)
# sys.exit()
# data_months *= 400 / 1e5
# data = cases_month_age()
# logger.debug(data_months)

# Parameters
state_0 = collect_state0()
# print (state_0.shape)
state_0 = pack_flat(state_0)
logger.debug("State 0 {}".format(state_0.shape))
warm_up = 100

# sys.exit("After collect state 0")
###################################################################################################################
#########################                            Run                         ##################################
###################################################################################################################

# Parameters
o = 4
phi = 0
f_top = 25
divi = 0.5
f1 = 10 * divi
f2 = 2 * divi
f3 = 2 * divi
f = np.concatenate((nums(f1, sc[0]), nums(f2, sc[1]), nums(f3, sc[2])))
# p = 1 / 100

# State 0
# state_0 = odeint(hetro_model, state_0, np.arange(0, warm_up * 365),
#                  args=(4, 0, 0.1, r_start - warm_up))  # [-1,:]
# x_old = reduce_time(np.arange(0, warm_up * 365), start=r_start - warm_up, step=step)
# y_old = unpack(state_0.T, *unpack_values)
# state_0 = state_0[-1, :]
# logger.info("state_0 collected")
# logger.debug("State 0: {}\n".format(state_0))

# Solve system
clk = clock()
RES = odeint(hetro_model, state_0, t_range,
             args=(o, phi, f, r_start))
logger.debug("*_" * 40)
logger.warning(" " * 40 + "TIME{:10.2f}".format(clock() - clk))
# # Results
# RES *= omega_ap
x = reduce_time(t_range, start=r_start, step=step)
y = unpack(RES.T, *unpack_values)
h = sum([i for i in y[:3]])
all = sum([i for i in y])
y.append(h)
y.append(all)
# print (y[3].sum(axis=0)[-100:])
# PLOT
names = ["Susceptible", "Vaccinated aP", "Vaccinated wP", "Infected Is", "Infected Ia", "Recovered", "Healthy", "All"]
draw_ages = [0, 1, -4, -3, -2, -1]
fig1, ax1 = draw_model(x, y[3:], names[3:], split=0, collapse=True, ages=draw_ages)
pop = np.genfromtxt('./data/demographics/birth_rate.csv', delimiter=',', usecols=[0, 2], skip_header=1)
ax1[-1].plot(pop[:, 0], 1000 * pop[:, 1], label="Real", c='k')
#
fig5, ax5 = draw_model(x, [100 * y[i] / y[-1].sum(axis=0) for i in range(3,8)], names[3:], split=0, collapse=True, ages=draw_ages)
# pop = np.genfromtxt('./data/demographics/birth_rate.csv', delimiter=',', usecols=[0, 2], skip_header=1)
# ax1[-1].plot(pop[:, 0], 1000 * pop[:, 1], label="Real", c='k')
# fig3, ax3 = draw_model(x, y[2:3], names[2:3], split=False, collapse=False, ages=draw_ages)
#
# fig2, ax2 = plt.subplots()
# ax2.plot(x, y[3].sum(axis=0)/y[-1].sum(axis=0), label='S')
# ax2.plot(x, y[4].sum(axis=0)/y[-1].sum(axis=0), label='A')
# ax2.legend()

fig3, ax3 = plt.subplots()
Ii = y[3].sum(axis=0)
ax3.plot(x, new_cases(Ii, gamma_s), label="New Cases Function")
# ax3.plot(x, Ii * gamma_s, label="Cases * {:.2f}".format(gamma_s))
# ax3.scatter(x, Ii * 60 / 70 , label="Model Untouched", c='k')
ax3.scatter(months, data.sum(axis=1) / p, label="Data")
ax3.legend()
ax3.set_title("NEW SICK {}".format(1 / N))

# fig4, ax4 = plt.subplots()
# ax4.plot(x, y[0].sum(axis=0))
# ax4.plot(x_old, y_old[0].sum(axis=0))

# fig, axs = plt.subplots()
# axs.plot(x, y[-1].sum(axis=0), label = "Sim")
# axs.plot(pop[:,0],1000*pop[:,1], label="Real")
# axs.legend()

plt.tight_layout()
plt.show()
