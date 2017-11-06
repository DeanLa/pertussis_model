# from scipy.integrate import odeint
# import numpy as np
# from numpy import cos, pi
import matplotlib.pyplot as plt
# import time
from pertussis import *

# def fix(y, r):
#     z = np.split(y, np.cumsum(sc)[:-1])
#     z = [yi.sum(axis=0) for yi in z]
#     z = [np.sum(yi.reshape(-1, r), axis=1) for yi in z]
#     z = np.array(z)
#     return z
# Initial
r_start = 1929
r_end = 2014  # 2014 is end of data
step = 1 / N

data1, months = cases_monthly()
data2, data2n, years = cases_yearly()
print("***Data Shapes***")
print(months.shape, data1.shape)
print(years.shape, data2.shape)

# Parameters
state_0 = collect_state0(S0=0.8, Is0=0.0000001)

# sys.exit("After collect state 0")
###################################################################################################################
#########################                            Run                         ##################################
###################################################################################################################

# Parameters
# om = 3.7
# phi = 0.6
# f_top = 25
# divi = 0.004
# f1 = 3 * divi
# f2 = 1 * divi
# f3 = 1 * divi
# # rho = 0.05 + max(f1, f2, f3)
# rho = 0.05
# e = 0.66
# om, phi, rho, f1, f2, f3, e = 3.81711623, 4.1158975, 0.02531624, 0.00101965, 0.000102348, 0.000103108, 0.5
om, phi, rho, f1, f2, f3, e = 4.4169, 3.6979, 20, 0.3143, 0.3682, 0.3319, 0.864
f = np.concatenate((nums(f1, sc[0]), nums(f2, sc[1]), nums(f3, sc[2])))
print('***Initial Params***')
print(rho, om, phi, f1, f2, f3)
# Solve system
clk = clock()
years_prior = 10
print (state_0)
y_0 = difference_model(state_0, r_start - years_prior, r_start,
                       rho, om, phi, f, e,
                       r=20, full_output=True)
print(y_0[3].shape)
state_0 = [yi[:, -1] for yi in y_0]
print (state_0)
print (([any (yi<0) for yi in state_0]))
print (np.where([any (yi<0) for yi in state_0]))

r = 3
y = difference_model(state_0, r_start, r_end,
                     rho, om, phi, f, e,
                     r=r, full_output=True)

print("TIME: ", clock() - clk)

############################################## PLOTTING ###########################
A = y[7].sum(axis=0)
A_0 = y_0[7].sum(axis=0)
names = ["Susceptible", "Vaccinated aP", "Vaccinated wP", "Infected Is", "Infected Ia", "Recovered", "Healthy", "All",
         "New"]
order = [6, 5, 3, 4, 7]
x = np.arange(r_start, r_end, N / r)
Y = [y[i] for i in order]
Y_0 = [y_0[o].sum(axis=0) for o in order]
names = [names[i] for i in order]
draw_ages = range(9)  # [0, 1, 2, -1, -2]
pop = np.genfromtxt('./data/demographics/birth_rate.csv', delimiter=',', usecols=[0, 2], skip_header=1)

# fig0, ax0 = plt.subplots()
# tmp_R = y[5][9,:]
# tmp_A = y[7][9,:]
# ax0.plot(x, tmp_R/ tmp_A)
# plt.show()
# exit()
###### Draw Figure 1#### Normalized
fig1, ax1 = draw_model(x, [yi / A for yi in Y], names, split=2, collapse=True, ages=draw_ages)
fig1.suptitle("Normalized", fontsize=20)
# print(A_0)

ax1[4].plot(pop[:, 0], 1000 * pop[:, 1], label="Real", c='k')
normalizer = A_0
for i, yo in enumerate(Y_0):
    ax1[i].plot(np.linspace(r_start - years_prior, r_start, len(yo)), yo / normalizer)

###### Draw Figure 2#### TOTALS
fig2, ax2 = draw_model(x, [yi / 1 for yi in Y], names, split=2, collapse=True, ages=draw_ages)
fig2.suptitle("Total", fontsize=20)
for axt in ax2:
    axt.set_ylim(-1,5)
# print(A_0)
ax2[4].plot(pop[:, 0], 1000 * pop[:, 1], label="Real", c='k')  # Real Population
normalizer = 1
for i, yo in enumerate(Y_0):
    ax2[i].plot(np.linspace(r_start - years_prior, r_start, len(yo)), yo / normalizer)

##### Draw Figure 3####
z = fix(y[-1], r)
z = z.sum(axis=0)  # Sum to all Ages
z = np.sum(z.reshape(-1, 12), axis=1)
a = A[::12 * r]
z_0 = fix(y_0[-1], 12 * 20).sum(axis=0)[3:]
a_0 = A_0[::12 * 20][3:]
x = np.arange(r_start, r_end, 12 * r * N / r)
print(z.shape, x.shape)

# fig3, ax3 = plt.subplots(1, 2, figsize=(18, 8))
# ax = ax3[0]
# ax.scatter(x, z, c='r', s=50, marker='1')
# ax.plot(x, z, c='r', lw=0.2)
# ax.set_title("Total")
#
# ax = ax3[1]
# ax.scatter(x, z / a, c='r', s=50, marker='1')
# ax.plot(x, z / a, c='r', lw=0.2)
# ax.set_title("Normalized")
#
# pop = np.vstack(([[i, 1789.1] for i in [1951, 1952, 1953, 1954]], pop))
# pop = pop[:1998 - 2014]
#
#
# # a_0 = A_0[::12*20][:]
# ax3[0].scatter(np.linspace(r_start - years_prior, r_start, len(z_0)), z_0, c='b', s=50, marker='1')
# ax3[1].scatter(np.linspace(r_start - years_prior, r_start, len(z_0)), z_0 / a_0, c='b', s=50, marker='1')
#
# # DATA
# print(data1.shape)
#
# data1y = data1.sum(axis=0)
# data1y = np.sum(data1y.reshape(-1, 12), axis=1)
# print(data1y / p)
# data1y = data1y / p

# print(data1y)
# data1yn = data1y / A[-16:]
# data1v = [data1y, data1yn]
# data2v = [data2, data2n]
# print(data1y.shape)
# for i, d in enumerate(zip(data1v, data2v)):
#     ax = ax3[i]
#     ax.scatter(np.arange(1998, 2014, 1), d[0], c='y', s=50)
#     ax.scatter(years, d[1], c='g', s=50)
#     lims = ax.get_ylim()
#     ax.vlines([1957, 1948, 2002, 1951], 0, 1000, lw=0.2, linestyle='--')
#     ax.set_ylim(np.minimum(-5,lims))
# z = fix(y[-1], r)
# # print(z.shape)
# z = z[:, -192:]



###### Draw Figure 4####
# fig, axs = plot_monthly(months, data1, z)
# fig4, ax4 = plt.subplots(1, 2, figsize=(18, 8))
# z = fix(y[-1], r)
# print(z.shape)
# z = z[:, -192:]
# labels = ['0-1', '1-21', '21+']
# ax = ax4[0]
# for i, l in enumerate(labels):
#     ax.plot(months, z[i, :], label=l)
#     ax.scatter(months, data1[i, :])
# ax.legend()
#
# ax = ax4[1]
# ax.plot(months, z.sum(axis=0), color='grey')
# ax.scatter(months, data1.sum(axis=0), c='grey')

plt.tight_layout()
plt.show()
