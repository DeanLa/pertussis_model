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

###### Run Model

# Priors
# m1 = pm.Uniform('m1', 0, 1, value=0.1)
o = pm.Uniform('omega', 3, 6, value=4)
p = pm.Uniform('phi', 0, o + 0.1, value=2)
f1 = pm.Uniform('f1', 0, 1)
f2 = pm.Uniform('f2', 0, 1)
f3 = pm.Uniform('f3', 0, 1)


@pm.deterministic
def f(f1=f1, f2=f2, f3=f3):
    s1, s2 = 5, 5
    s3 = J - s1 - s2
    return np.concatenate((f1 * np.ones(s1),
                           f2 * np.ones(s2),
                           f3 * np.ones(s3)))


times = []


@pm.deterministic
def sim(o=o, p=p, f=f):
    # print('A ', end="")
    clk = clock()
    RES = odeint(hetro_model, state_0, t_range,
                 args=(o, p, f),
                 full_output=False)

    # print (RES.shape)
    # print(RES.sum(axis=1))
    RES = unpack(RES.T, *unpack_values)
    # print (RES[3].shape)
    res = reduce_year(RES[3].sum(axis=0))[1951 - t_start:2014 - t_start]
    # print (res.shape)
    print(clock() - clk)
    times.append(clock() - clk)
    return res


Y = pm.Normal('Y', mu=sim, tau=1, observed=True, value=data)

model = pm.Model([Y, o, p, sim, f, f1, f2, f3])
mcmc = pm.MCMC(model, db="ram")
mcmc.sample(iter=4000, burn=0)
times = np.array(times)
print ()
print(times.min(), times.mean(), times.max())
# print (mcmc.summary())
m_f = mcmc.trace('f')[:].mean()
m_o = mcmc.trace('omega')[:].mean()
m_p = mcmc.trace('phi')[:].mean()

clk = clock()
RES = odeint(hetro_model, state_0, t_range,
             args=(m_o, m_p, m_f))
print(clock() - clk)
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
fig4, ax4 = plot_stoch_vars(mcmc)
fig4.savefig('./img/mcmc{}.png'.format(clk))
# fig,ax = plt.subplots()
# ax.plot(x[20000:-1], y[3][20000:-1])
plt.tight_layout()
# plt.show()
