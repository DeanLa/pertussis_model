import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.integrate import odeint
from time import clock, sleep
from datetime import datetime as dt
from pprint import pprint
from pertussis import *

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
# TODO: Data Monthly
# data_month
# months

#######################
###### Run Model ######
#######################
i0 = pm.Uniform("Is0", 0, 0.1)
s0 = pm.Uniform("S0", 0.1, 1 - i0)


# State 0
@pm.deterministic
def state_0(S=s0, I=i0):
    state_0 = collect_state0(S, I)
    # state_0 = pack_flat(state_0)
    return state_0


# sleep(0.01)  # makes prints clearer

# Priors
o = pm.Uniform('omega', 3, 6, value=4)
p = pm.Uniform('phi', 0, o + 0.1, value=2)
f1 = pm.Uniform('f1', 0, 0.5)
f2 = pm.Uniform('f2', 0, 0.5)
f3 = pm.Uniform('f3', 0, 0.5)


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
    RES = odeint(hetro_model, pack_flat(state_0), t_range,
                 args=(o, p, f))

    # print (RES.shape)
    # print(RES.sum(axis=1))
    RES = unpack(RES.T, *unpack_values)
    # print (RES[3].shape)
    # res = reduce_year(RES[3].sum(axis=0))[1951 - t_start:2014 - t_start]
    # print (res.shape)
    print(clock() - clk)
    times.append(clock() - clk)
    return RES


@pm.deterministic
def mu1(RES=sim):
    res = reduce_year(RES[3].sum(axis=0))[1951 - t_start:2014 - t_start]
    return res


# TODO: Two sigmas
sigma1 = pm.Uniform('sigma1', 0, 0.5)
# sigma2 = pm.Uniform('sigma2', 0, sigma1/12)
Y1 = pm.Normal('Y1', mu=mu1, tau=1 / sigma1 ** 2, observed=True, value=data)
# Y2 = pm.Normal('Y2', mu=mu2, tau=1 / sigma2 ** 2, observed=True, value=data_monthly)

# TODO: Set weights

model = pm.Model([Y1, o, p, sim, f, f1, f2, f3, mu1, sigma1, s0, i0, state_0])
# TODO: Other Backend
mcmc = pm.MCMC(model, db="ram")
mcmc.sample(iter=5, burn=0)
times = np.array(times)
print()
print(times.min(), times.mean(), times.max())
# print (mcmc.summary())
t_tally = 0
m_f = mcmc.trace('f')[t_tally:].mean()
m_o = mcmc.trace('omega')[t_tally:].mean()
m_p = mcmc.trace('phi')[t_tally:].mean()
m_s = mcmc.trace('S0')[t_tally:].mean()
m_i = mcmc.trace('Is0')[t_tally:].mean()

# Show mean values fit
state_0 = collect_state0(m_s, m_i)
state_0 = pack_flat(state_0)
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
fig4.savefig('./img/mcmc{}.png'.format(dt.utcnow()).replace(':', '-'))
# fig,ax = plt.subplots()
# ax.plot(x[20000:-1], y[3][20000:-1])
plt.tight_layout()
plt.show()
