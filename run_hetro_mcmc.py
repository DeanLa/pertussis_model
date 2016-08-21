import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from scipy.integrate import odeint
from time import clock, sleep
from datetime import datetime as dt
from pprint import pprint
import pickle
from copy import copy

from pertussis import *

# J = 12
# print(J)
# Initial
r_start = 1920
r_end = 2015
step = 1 / N
t_end = expand_time(r_end, start=r_start, step=step)
t_start = expand_time(r_start, start=r_start, step=step)
t_range = np.arange(t_start, t_end, 1)
vars = copy(locals())
times = []
# Data
report_rate = np.append(nums(250, 7), nums(400, J - 7))  # Danny Cohen paper. pg3
data_years, years = cases_yearly()  # per 1e5
data_years *= 400 / 1e5
data_months, months = cases_monthly()  # per 1e5
data_months *= 400 / 1e5
data = cases_month_age()

# Parameters
state_0 = collect_state0()
state_0 = pack_flat(state_0)

#######################
###### Run Model ######
#######################

# Priors

omega = pm.Uniform('omega', 3, 6)
phi = pm.Uniform('phi', 0, omega + 0.1)
f_top = 25
f1 = pm.Uniform('f1', 0, f_top)
f2 = pm.Uniform('f2', 0, f_top)
f3 = pm.Uniform('f3', 0, f_top)

p = pm.Uniform('p', 1 / 400, 1 / 60)


@pm.deterministic
def f(f1=f1, f2=f2, f3=f3):
    s1, s2 = 4, 3
    s3 = J - s1 - s2
    return np.concatenate((nums(f1, s1), nums(f2, s2), nums(f3, s3)))


@pm.deterministic(trace=False)
def sim(omega=omega, phi=phi, f=f):
    clk = clock()
    res = odeint(hetro_model, pack_flat(state_0), t_range,
                 args=(omega, phi, f, r_start))

    res = unpack(res.T, *unpack_values)
    print(clock() - clk)
    times.append(clock() - clk)
    return res


@pm.deterministic
def mu(sim=sim, omega=omega, phi=phi, f=f):
    x = reduce_time(t_range, start=r_start, step=step)
    y = new_cases(x, sim[0], sim[1], sim[2], sim[3], sim[4], f=f, omega=omega, phi=phi)
    start_ix = (1998 - r_start) * 12
    end_ix = (2014 - r_start) * 12
    # print (end_ix - start_ix)
    res = reduce_month(y)[:, start_ix:end_ix]
    return res


# Y = pm.Normal('Y', mu=mu1, tau=1 / sigma1 ** 2, observed=True, value=data)
Y = pm.Binomial('Y', n=mu, p=p, observed=True, value=data)

# TODO: Other Backend
model = pm.Model([Y, sim, f, mu, omega, phi, f1, f2, f3, ])
mcmc = pm.MCMC(model, db="ram")
gclk = clock()
mcmc.sample(iter=10, burn=0)  #######################################################################################
print ("Global Time ",clock()-gclk)

# times = np.array(times)
# print(times.min(), times.mean(), times.max())

# Traces
t_tally = 0
m_f = mcmc.trace('f')[t_tally:].mean(axis=0)
m_omega = mcmc.trace('omega')[t_tally:].mean()
m_phi = mcmc.trace('phi')[t_tally:].mean()
tr_mu = mcmc.trace('mu', chain=None)[:]
m_mu = tr_mu.mean(axis=0)
m_p = mcmc.trace('p', chain=None)[:].mean(axis=0)

# Show mean values fit
clk = clock()
RES = odeint(hetro_model, state_0, t_range,
             args=(m_omega, m_phi, m_f, r_start))
print(clock() - clk)
# # Results
x = reduce_time(t_range, start=r_start, step=step)
y = unpack(RES.T, *unpack_values)
h = sum([i for i in y[:3]])
all = sum([i for i in y])
y.append(h)
y.append(all)

# PLOT
figs = []
fig, axs = mu_chart(tr_mu * m_p, data)
figs.append(fig)

fig1, ax1 = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=True)
figs.append(fig1)

fig2, ax2 = draw_model(x, y[0:3], ["Susceptible", "Vaccinated ap", "Vaccinated wp"], split=False, collapse=True)
figs.append(fig2)

fig3, ax3 = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=True)
ax3[0].scatter(years, data_years / 12)
ax3[0].scatter(months, data_months, c='green')
ax3[0].set_xlim([2000, 2015])
ax3[0].set_ylim(0, 0.05)
figs.append(fig3)

plt.tight_layout()
path = write_report(vars, x, y, figs, mcmc)
# with open (path+"mcmc.pickle", 'wb') as pickle_file:
#     pickle.dump(mcmc.trace(), pickle_file)
# plt.show()
