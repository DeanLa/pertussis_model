from scipy.integrate import odeint
import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt
import time
from pertussis import *
from pertussis.model import hetro_model
from pertussis.params.hetro_model import collect_state0
from pertussis.charts import draw_model
import pymc as pm
from time import clock
from pprint import pprint

plt.style.use('ggplot')

# State 0
state_0 = collect_state0()
pprint(state_0)
state_0 = pack_flat(state_0)
time.sleep(0.01)  # makes prints clearer
print ("\n\n\n")
pprint(state_0.sum())

# # Initial Parameters
t_start = 1948
t_end = 2024
step = 365
t_end = expand_time(t_end, step=step)
t_range = np.arange(t_start, t_end + 0, 1)

# # Data
# data = np.genfromtxt('./data/yearly.csv', delimiter=',', skip_header=1)[:, 1]
# data /= 100000
# years = np.genfromtxt('./data/yearly.csv', delimiter=',', skip_header=1)[:, 0]
#
# ###### Run Model
#
# # Priors
# m1 = pm.Uniform('m1', 0, 1, value=0.1)
# m2 = pm.Uniform('m2', 0, 0.8, value=0.1)
# omega = pm.Uniform('omega', 0, 16, value=4)
# phi = pm.Uniform('phi', 0, omega + 0.1, value=2)
#
# Y = pm.Normal('Y', mu=sim, tau=1, observed=True, value=data)
#
# mcmc = pm.MCMC([m1, m2, omega, phi, sim, Y], db="ram")
# mcmc.sample(iter=2, burn=0)
# print (mcmc.summary())
# m_m1 = mcmc.trace('m1')[:].mean()
# m_m2 = mcmc.trace('m2')[:].mean()
# m_omega = mcmc.trace('omega')[:].mean()
# m_phi = mcmc.trace('phi')[:].mean()
#
# RES = odeint(vaccine_model, state_0, t_range,
#              args=(step, m_m1, m_m2, m_omega, m_phi))
# # # Results
# x = reduce_time(t_range, start=t_start, step=step)
# y = unpack(RES.T, 1, 6, 4, 1, 1, 1)
# h = y[0] + y[1].sum(axis=0) + y[2].sum(axis=0)
# y.append(h)
# #
# fig2, ax2 = draw_model(x, y[0:3], ["Susceptible", "Vaccinated ap", "Vaccinated wp"], split=False)
# fig1, ax1 = draw_model(x, y[3:7], ["Infected s", "Infected Ia", "Recovered", "Healthy"], split=0)
# ax1[0].scatter(years, data)
# # # fig,ax = plt.subplots()
# # # ax.plot(x[20000:-1], y[3][20000:-1])
# plt.tight_layout()
# plt.show()
