from scipy.integrate import odeint
import numpy as np
# from numpy import cos, pi
import matplotlib.pyplot as plt
from pertussis import *
from pertussis.model import hetro_model
from pertussis.params.hetro_model import collect_state0
from pertussis.charts import draw_model
import pymc as pm
from time import clock, sleep
from pprint import pprint

plt.style.use('ggplot')

# State 0
state_0 = collect_state0()
# pprint(state_0)
state_0 = pack_flat(state_0)
sleep(0.01)  # makes prints clearer
# print ("\n")
# print(state_0.sum())


# # Initial Parameters
t_start = 1948
t_end = 2024
step = 1 / N
print (step)
t_start = expand_time(t_start, step=step)
t_end = expand_time(t_end, step=step)
t_range = np.arange(t_start, t_end + 0, 1)
print (t_range)

m_m1 = 0.1
m_omega = 4
m_phi = 0.25
#
clk = clock()
RES = odeint(hetro_model, state_0, t_range,
             args=(step, m_m1, m_omega, m_phi))
print (clock()-clk)
# # # Results
x = reduce_time(t_range, start=t_start, step=step)
y = unpack(RES.T, *unpack_values)
h = y[0] + y[1] + y[2]
y.append(h)
# #
fig2, ax2 = draw_model(x, y[0:3], ["Susceptible", "Vaccinated ap", "Vaccinated wp"], split=False, collapse=False)
fig1, ax1 = draw_model(x, y[3:7], ["Infected s", "Infected Ia", "Recovered", "Healthy"], split=0, collapse=True)
# ax1[0].scatter(years, data)
# # # fig,ax = plt.subplots()
# # # ax.plot(x[20000:-1], y[3][20000:-1])
plt.tight_layout()
plt.show()
