import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.integrate import odeint
from time import clock, sleep
from pprint import pprint
from pertussis import *

# State 0
state_0 = collect_state0()
state_0 = pack_flat(state_0)
sleep(0.01)  # makes prints clearer


# # Initial Parameters
t_start = 1948
t_end = 2015
step = 1 / N
# t_start = expand_time(t_start, step=step)
t_end = expand_time(t_end, step=step)
t_range = np.arange(t_start, t_end + 0, 1)

# # Data
data = np.genfromtxt('./data/yearly.csv', delimiter=',', skip_header=1)[:, 1]
data /= 100000
years = np.genfromtxt('./data/yearly.csv', delimiter=',', skip_header=1)[:, 0]

m_f = np.ones(15) * 0.6
m_o = 4
m_p = 2
# plot_stoch_vars(mcmc)
clk = clock()
RES = odeint(hetro_model, state_0, t_range,
             args=(m_o, m_p, m_f))
print(clock() - clk)
# print(RES.sum(axis=1))
# print(RES.sum(axis=1).shape)
# # Results
x = reduce_time(t_range, start=t_start, step=step)
y = unpack(RES.T, *unpack_values)
h = sum([i for i in y[:3]])
all = sum([i for i in y])
y.append(h)
y.append(all)

# fig1, ax1 = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=False)
fig3, ax3 = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=True)
fig2, ax2 = draw_model(x, y[0:3], ["Susceptible", "Vaccinated ap", "Vaccinated wp"], split=False, collapse=True)
ax3[0].scatter(years, data)
# fig,ax = plt.subplots()
# ax.plot(x[20000:-1], y[3][20000:-1])
plt.tight_layout()
plt.show()
