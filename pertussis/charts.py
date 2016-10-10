import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
from pertussis import a_l, a_u, J

plt.style.use('ggplot')


def draw_no_split(x, y):
    # l = y.shape[1]
    l = len(y)
    fig, axs = plt.subplots(l, figsize=(12, 8))

    return fig, axs


def draw_split(x, y, which):
    l = len(y)
    fig = plt.figure(figsize=(14, 7))
    gs = plt.GridSpec(4, 2)
    axs = []
    axs.append(plt.subplot(gs[2, 0]))
    axs.append(plt.subplot(gs[2, 1]))
    axs.append(plt.subplot(gs[0:2, 0:2]))  # Number 2 - Main Graph
    axs.append(plt.subplot(gs[3, 0]))
    axs.append(plt.subplot(gs[3, 1]))
    if which != 2:
        axs[2], axs[which] = axs[which], axs[2]
    return fig, axs


def draw_model(x, y, labels=None, split=None, collapse=False, ages = np.arange(0,J,1)):
    l = len(y)
    if l > 5:
        split = False
    if type(split) is int:
        assert l <= 5, "Must have at most 5 variables"
        fig, axs = draw_split(x, y, which=split)
    else:
        fig, axs = draw_no_split(x, y)
    for i in range(l):
        try:
            ax = axs[i]
        except:
            ax = axs
        z = y[i][ages, :]
        if collapse:
            z = y[i][:, :]
            z = z.sum(axis=0)
        lines = ax.plot(x, z.T, lw=1.3)
        try:
            ax.set_title(labels[i])
        except:
            ax.set_title("")
        ub = max(0.00001, z.max() * 1.05)
        # ub = 1
        ax.set_ylim([0, ub])
        # ax.set_xlim(left=1948)#, right = 2010)
        ax.vlines([1957, 2002, 1948], 0, ub, linestyle='--')
        if l <= 3:
            ax.legend(lines, ["{:.2f}-{:.2f}".format(a_l[j], a_u[j]) for j in ages], loc='lower left')
    if split:
        axs[split].legend(lines, [j + 1 for j in range(len(lines))], loc='lower left')

    return fig, axs


# MCMC
def plot_stoch_vars(mcmc, which=None, exclude=None):
    # Plot the variables
    # Initialize data frame for later analysis
    assert isinstance(mcmc, pm.MCMC), "mcmc should be MCMC object"
    if which:
        stoch = sorted(which)
    else:
        stoch = sorted([str(v) for v in mcmc.stochastics])
    if exclude:
        stoch.remove(exclude)
    from scipy.stats.mstats import mquantiles

    tr_len = mcmc.trace(stoch[0])[:, None].shape[0]
    # Height is determined by number of variables
    h = 3 * len(stoch)
    fig, axs = plt.subplots(len(stoch), 2, figsize=(16, h))

    for i, tr in enumerate(stoch):
        try:
            # print str(tr)
            # Prepare Values
            tr_val = mcmc.trace(tr)[:, None]
            quants = mquantiles(tr_val, prob=[0.025, 0.25, 0.5, 0.75, 0.975])

            # Plot
            ### Left: Histogram
            h = axs[i, 0].hist(tr_val, histtype='stepfilled',
                               bins=50, label=str(tr), alpha=0.6)
            m = max(h[0])
            axs[i, 0].fill_betweenx([0, m + 5], quants[0], quants[-1], color='lightgreen')
            axs[i, 0].fill_betweenx([0, m + 5], quants[1], quants[-2], color='darkgreen')
            axs[i, 0].set_title("{var} = {value:.4f} [{lci:.4f}, {hci:.4f}] ({tmp:.4f})".format(var=str(tr),
                                                                                                tmp=tr_val.mean(),
                                                                                                value=quants[2],
                                                                                                lci=quants[0],
                                                                                                hci=quants[-1]))
            axs[i, 0].set_xticks(axs[i, 0].get_xticks()[::2])  # X 0
            axs[i, 0].set_yticks(axs[i, 0].get_yticks()[::2])  # Y 0
            axs[i, 0].set_ylim([0, m + 5])
            # print axs[i,1].get_yticks()[::2]
            ### Right: Trace
            l = range(len(tr_val))
            axs[i, 1].plot(l, tr_val)
            axs[i, 1].set_xticks(axs[i, 1].get_xticks()[::2])  # X 1
            axs[i, 1].set_yticks(axs[i, 1].get_yticks()[::2])  # Y 1
            axs[i, 1].set_xlim([0, l[-1]])

        except:
            print(str(tr), " excluded")
    # fig.set_tight_layout
    fig.suptitle("Posterior Distribution <|> Convergence")
    return fig, axs


def mu_chart(mu, data):
    from pertussis import sc_ages
    from scipy.stats.mstats import mquantiles
    n_iterations, n_months, n_groups = mu.shape # Assumes (iteration, month, group)
    x = 1998 + np.arange(0, n_months, 1) / 12
    fig, axs = plt.subplots(3, int(np.ceil(n_groups/3)), figsize=(16, 16)) # Create J axes with 4 in a row
    axs = np.hstack(axs)
    for i in range(n_groups):
        ax = axs[i]
        q_mu = mu[:, :, i] # Takes group and all its lines
        quants = mquantiles(q_mu, prob=[0.05, 0.95], axis=0) #5% 95% quantiles
        ax.fill_between(x, quants[0], quants[1], color='red', alpha=0.3, label="95% CI")
        ax.plot(x, q_mu.mean(axis=0), label='Model')
        # ax.scatter(x, data[:, i], label="Data")
        ax.plot(x, data[:, i], '--', label="Data")
        ax.set_title('Age: {:.2f} - {:.2f}'.format(sc_ages[i], sc_ages[i+1]))
    axs[0].legend(loc='best')

    return fig, axs
