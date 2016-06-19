import matplotlib.pyplot as plt
import pickle
from pertussis import *

with open('./data/x.p','rb') as fl:
    x, y, years, data = pickle.load(fl)

# print (y[0])

# fig1, ax1 = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=False)
fig3, ax3 = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=True)
fig2, ax2 = draw_model(x, y[0:3], ["Susceptible", "Vaccinated ap", "Vaccinated wp"], split=False, collapse=False)
ax3[0].scatter(years, data)
# fig4, ax4 = plot_stoch_vars(mcmc)
# fig4.savefig('./img/mcmc{}.png'.format(dt.utcnow()).replace(':', '-'))
# fig,ax = plt.subplots()
# ax.plot(x[20000:-1], y[3][20000:-1])
plt.tight_layout()
plt.show()