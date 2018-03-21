from pertussis import *

from pprint import pprint
from time import clock, sleep
from scipy.integrate import odeint
from copy import copy
import numpy as np
import matplotlib.pyplot as plt


def chain_tally(mcmc, tally):
    if tally < len(mcmc['accepted']):
        mcmc['tally'] = tally
    else:
        mcmc['tally'] = len(mcmc['chain']) - 1


def chain_ll(mcmc):
    tally = mcmc['tally']
    return mcmc['ll'][tally:, :]


def mle_values(mcmc, with_min=True):
    '''return max_le, mle_place, min_le'''
    tally = mcmc['tally']
    ll = mcmc['ll'][tally:, :]
    # max_le = mcmc['max_likelihood']
    max_le = np.max(ll)
    try:
        place = np.where(ll[:, 0] == max_le)[0][0] + tally
    except:
        place = 0
    min_le = np.min(np.where(ll >= -99999, ll, 0))
    return max_le, place, min_le


def likelihood_progression(mcmc, zoom=None, tally=0, ax=None):
    tally = mcmc['tally']
    ll = mcmc['ll'][tally:, :].copy()
    mle, mle_place, min_le = mle_values(mcmc, True)
    l = len(mcmc['chain']) - tally
    xaxis = tally + np.arange(len(ll))
    # MLE
    best_vals = mcmc['chain'][mle_place, :]
    if not zoom:
        print("MLE: {:.2f} at {} with values: \n {}".format(mle, mle_place, best_vals), sep=",")

    #
    infs = np.where(ll[:, 1] < -9999)[0] + tally
    if ax:
        fig = ax.figure.canvas
    else:
        fig, ax = plt.subplots(figsize=(16, 8))
    # ax.plot(ll, lw=0.2)
    ax.plot(xaxis, mcmc['ll'][tally:, 0], label='Current Value', zorder=100)
    ax.plot(xaxis, mcmc['ll'][tally:, 1], alpha=0.5, label='Proposed')
    # ax.plot(mcmcs[1]['ll'][tally:,0])
    # ax.plot(mcmcs[2]['ll'][tally:,0])
    # MLE
    ax.hlines(mle, tally, xaxis[-1], linestyles='--')
    # 500 runs
    ax.vlines(xaxis[::500], min_le, mle, linestyles='--', alpha=0.2)
    # infs rug
    ax.vlines(infs, min_le, min_le + 10, lw=0.051)
    ax.scatter(mle_place, mle, s=300)
    title = 'Likelihood Progreesion'
    if zoom:
        ax.set_xlim(mle_place - zoom, mle_place + zoom)
        ll_zoom = mcmc['ll'][mle_place - zoom:mle_place + zoom, :]
        ax.set_ylim(1.05 * ll_zoom.min(), 0.95 * ll_zoom.max())
        title += ' Zoom: {}'.format(zoom)
    ax.set_title(title)
    ax.legend()

    # plt.show()
    return fig, ax


def chain_summary(mcmc):
    ll = chain_ll(mcmc)
    fig, axs = plt.subplots(2, 2, figsize=(24, 12))
    tally = mcmc['tally']

    # Acceptence rate
    ax = axs[0, 0]
    ax.plot(mcmc['rates'][1:], zorder=100)
    ax.set_title('$\\alpha$ Acceptance Rate')
    rates_mean = mcmc['rates'][-5:].mean()
    lims = ax.get_xlim()
    ax.hlines(mcmc['accept_hat'], lims[0], lims[1], linestyles='--', alpha=0.5, label="Threshold")
    ax.hlines(rates_mean, lims[0], lims[1], linestyles='-', alpha=0.5, label="Mean")
    ax.set_xlim(left=0)
    ax.legend()
    # Scaling Factor
    ax = axs[1, 0]
    ax.plot(np.arange(1, len(mcmc['scaling_factor']) + 1), mcmc['scaling_factor'])
    ax.set_title('$\lambda$ Scaling Factor')
    # likelihood difference
    ax = axs[1, 1]
    ldiff = ll[tally:, 0] - ll[tally:, -1]
    ldiff_finite = ldiff[np.abs(ldiff) < 1000]
    # Number of -inf
    ax.hist(ldiff_finite, bins=50)
    ax.set_title('L difference, INFS = {:.5f}'.format((ll[tally:, 1] < -9999).mean()))
    ax.vlines(ldiff_finite.mean(), 0, 70)
    plt.show()


def plot_chains(mcmc, dists=None, ax=None, fig=None, multi_chain=False):
    from scipy.stats import gaussian_kde as kde
    from scipy.stats.mstats import mquantiles
    tally = mcmc['tally']
    # mcmc = mcmc0
    chain = mcmc['chain'][tally:, :]
    guesses = mcmc['guesses'][tally:, :]
    titles = mcmc['names']
    titles = [['$\Omega$', '$\phi$', r'$\rho$', '$f_1$', '$f_2$', '$f_3$'][i] for i in mcmc['active_params']]
    l = len(mcmc['chain']) - tally

    if fig:
        axs = np.array(fig.get_axes()).reshape(len(titles), 2)
    else:
        fig, axs = plt.subplots(len(titles), 2, figsize=(18, 21))
        axs[0, 0].set_title("Chain")
        axs[0, 1].set_title("Distribution")
        # axs[0, 2].set_title("Gelman Rubin")
    for i, name in enumerate(titles):
        if multi_chain:
            ch = chain[:, i]
        else:
            ch = guesses[:, i]
        # LEFT - Chains
        # Limits
        if dists:
            a, b = dists[mcmc['active_params'][i]].args
            a, b = a, a + b
        else:
            a, b = min(ch), max(ch)
        qs = mquantiles(chain[:,i], [0.025,0.5,0.975])
        print (qs)
        a *= 0.9
        b *= 1.1
        ax = axs[i, 0]
        if multi_chain:
            my_alpha = 1 if mcmc['active'] else 0.6
            ax.plot(tally + np.arange(l), chain[:, i], label=mcmc['name'], alpha=my_alpha,
                    zorder=5 + 1 / (10 * my_alpha))  # , label = 'chain {}'.format(j))
        else:
            ax.plot(tally + np.arange(l), guesses[:, i],
                    color='steelblue', ls='--', alpha=0.5, label="Proposed")
            ax.plot(tally + np.arange(l), chain[:, i],
                    color='red', label="Accepted")  # , label = 'chain {}'.format(j))
            lims = mcmc['tally'], len(mcmc['chain'])
            ax.hlines(qs,*lims,lw=1,linestyles='--', zorder=10)
            ax.set_xlim(*lims)
            ax.set_yticks(qs)

        ax.legend()

        # MIDDLE - Distribution
        try:
            if mcmc['active']:
                density = kde(chain[:, i])
                xs = np.linspace(a, b, 50)
                axs[i, 1].plot(xs, density(xs))
        except:
            print("{} Can't KDE yet on {}".format(mcmc['name'], name))
        axs[i,1].set_ylabel(name, rotation=0, fontsize=30,)
        c, d = axs[i, 1].get_xlim()
        a = min(a, c)
        b = max(b, d)
        axs[i, 1].set_xlim(a, b)
        # RIGHT - Gelman Rubin
        # if plot_gr:
        #     axs[i, 2].plot(mcmc['gelman_rubin'][2:, i], color='green')
        #     axs[i, 2].hlines(1, 0, len(mcmc['gelman_rubin']))
        #     axs[i, 2].hlines(1.1, 0, len(mcmc['gelman_rubin']), linestyles='--')

    plt.tight_layout()
    return fig, axs


def likelihood_progression_multi(mcmcs):
    fig, ax = plt.subplots(figsize=(16, 8))
    for mc in mcmcs:
        tally = mc['tally']
        l = len(mc['chain']) - tally
        mle, mle_place, min_le = mle_values(mc, True)
        best_vals = mc['chain'][mle_place, :]
        print("{} - MLE: {:.2f} at {} (of {})with values: \n {}".format(mc['name'], mle, mle_place,
                                                                        len(mc['chain']), best_vals), sep=",")
        my_alpha = 1 if mc['active'] else 0.8
        ax.plot(tally + np.arange(l), mc['ll'][mc['tally']:, 0], label=mc['name'], alpha=my_alpha,
                zorder=12 - 10 * my_alpha)
        ax.scatter(mle_place, mle, s=300, zorder=1 + 10 * my_alpha, edgecolor='k', linewidth=2, alpha=my_alpha)
    ax.legend()
    ax.set_title('Likelihood Progreesion')
    # ax.set_xlim((0,1000))
    return fig, ax

def policy_comparison(df, colors, ax=None):
    import seaborn as sns
    fontdict = {'fontsize': 12}
    policy_names = df.columns
    if ax:
        fig = ax.figure.canvas
    else:
        fig, ax = plt.subplots(figsize=(16, 16))
    sns.boxplot(data = df, ax=ax, orient="h", palette=colors, linewidth=1)
    win_ratio = (df > 0).mean(axis=0).values
    ax.set_yticklabels(policy_names, fontdict=fontdict)
    ax.vlines(0, 0, len(policy_names), linestyles='--', alpha=0.3)

    lims = list(ax.get_xlim())
    lims[0] *= 1
    ax.hlines(np.arange(len(policy_names)), *lims, linestyles='--', alpha=0.3, zorder=-5, lw=0.5)
    ax.set_xlim(*lims)
    for i, r in enumerate(win_ratio):
        if (r<0.0001) or (r>0.9999): continue
        text_color = 'g' if r>0.8 else 'k'
        ax.annotate("{:.2f}%".format(100*r), xy=(lims[0], i-0.1), fontsize=16, color=text_color)

    plt.tight_layout()
    return fig, ax

def spline(arrx, arry, interval=0.05, med=True):
    l = np.arange(0,1,0.01)
    u = np.arange(interval,1+interval,0.01)
    # print(len(u),len(l))
    line = []
    for a,b in zip (l,u):
        g = arry[(arrx>=a) & (arrx<=b)]
        # line.append(g.mean())
        if med:
            line.append(np.median(g))
        else:
            line.append(np.mean(g))
    large = np.median(arry[(arrx >= 0.0) & (arrx <= 0.03)])
    small = np.median(arry[(arrx>=0.97) & (arrx<=1)])
    print ("{} - {} = {}: reduction {}".format(large,small,large-small, small/large))
    return (l+u)/2, line

def random_scatter(df, var):
    fig, ax = plt.subplots()
    ax.scatter(df[var], df.ll, alpha=0.005, s=10, c='C2')
    ticks = np.arange(0, 1 + 0.001, 0.05)
    ax.set_xticks(ticks)
    # ax.set_xticklabels(ticks * 100 + 10)
    ax.hlines(df.ll.max(), 0, 1, lw=0.1)
    return fig, ax