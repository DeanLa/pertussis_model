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
        mcmc['tally'] = -1


def chain_ll(mcmc):
    tally = mcmc['tally']
    return mcmc['ll'][tally:, :]


def mle_values(mcmc, with_min=True):
    '''return max_le, mle_place, min_le'''
    tally = mcmc['tally']
    ll = mcmc['ll'][tally:, :]
    # max_le = mcmc['max_likelihood']
    max_le = np.max(ll)
    place = np.where(ll[:, 0] == max_le)[0][0]
    min_le = np.min(np.where(ll >= -99999, ll, 0))
    return max_le, place, min_le


def likelihood_progression(mcmc, zoom=None, tally=0, ax=None):
    tally = mcmc['tally']
    ll = mcmc['ll'][tally:, :].copy()
    mle, mle_place, min_le = mle_values(mcmc, True)

    # MLE
    best_vals = mcmc['chain'][mle_place, :]
    if not zoom:
        print("MLE: {:.2f} at {} with values: \n {}".format(mle, mle_place, best_vals), sep=",")

    #
    infs = np.where(ll[:, 1] < -9999)[0]
    if ax:
        fig = ax.figure.canvas
    else:
        fig, ax = plt.subplots(figsize=(16, 8))
    # ax.plot(ll, lw=0.2)
    ax.plot(mcmc['ll'][tally:, 0], label='Current Value', zorder=100)
    ax.plot(mcmc['ll'][tally:, 1], alpha=0.5, label='Proposed')
    # ax.plot(mcmcs[1]['ll'][tally:,0])
    # ax.plot(mcmcs[2]['ll'][tally:,0])
    # MLE
    ax.hlines(mle, 0, len(mcmc['ll']) - tally, linestyles='--')
    # 500 runs
    ax.vlines(np.arange(1, len(mcmc['ll']) - tally, 500), min_le, mle, linestyles='--', alpha=0.2)
    # infs rug
    ax.vlines(infs, min_le, min_le + 10, lw=0.051)
    ax.scatter(mle_place, mle, s=300)
    title = 'Likelihood Progreesion'
    if zoom:
        ax.set_xlim(mle_place - zoom, mle_place + zoom)
        ll_zoom = ll[mle_place - zoom:mle_place + zoom, :]
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


def plot_chains(mcmc, dists=None, ax=None, plot_gr=False, fig=None, multi_chain=False):
    from scipy.stats import gaussian_kde as kde
    tally = mcmc['tally']
    # mcmc = mcmc0
    chain = mcmc['chain'][tally:, :]
    guesses = mcmc['guesses'][tally:, :]
    titles = mcmc['names']
    titles = [['$\Omega$', '$\phi$', r'$\rho$', '$f_1$', '$f_2$', '$f_3$'][i] for i in mcmc['active_params']]

    if fig:
        axs = np.array(fig.get_axes()).reshape(-1, 3)
    else:
        fig, axs = plt.subplots(len(titles), 3, figsize=(18, 21))
        axs[0, 0].set_title("Chain")
        axs[0, 1].set_title("Chain")
        axs[0, 2].set_title("Gelman Rubin")
    for i, name in enumerate(titles):
        ch = guesses[:, i]
        # LEFT - Chains
        # Limits
        if dists:
            a, b = dists[i].args
            a, b = a, a + b
        else:
            a, b = min(ch), max(ch)
        a *= 0.9
        b *= 1.1
        ax = axs[i, 0]
        if multi_chain:
            my_alpha = 1 if mcmc['active'] else 0.4
            ax.plot(chain[:, i], label=mcmc['name'], alpha = my_alpha, zorder=10*my_alpha)  # , label = 'chain {}'.format(j))
        else:
            ax.plot(guesses[:, i], color='steelblue', ls='--', alpha=0.5, label="Proposed")
            ax.plot(chain[:, i], color='red', label="Accepted")  # , label = 'chain {}'.format(j))
        ax.set_ylabel(name, rotation=0, fontsize=16)
        #     ax.set_ylim(a,b)
        ax.set_xlim(0, len(chain[:, i]))
        ax.legend()

        # MIDDLE - Distribution
        try:
            density = kde(chain[:, i])
            xs = np.linspace(a, b, 50)
            axs[i, 1].plot(xs, density(xs))
        except:
            print ("{} Can't KDE yet".format(mcmc['name']))
        # RIGHT - Gelman Rubin
        if plot_gr:
            axs[i, 2].plot(mcmc['gelman_rubin'][2:, i], color='green')
            axs[i, 2].hlines(1, 0, len(mcmc['gelman_rubin']))
            axs[i, 2].hlines(1.1, 0, len(mcmc['gelman_rubin']), linestyles='--')

    plt.tight_layout()
    return fig, axs

def likelihood_progression_multi(mcmcs):
    fig, ax = plt.subplots(figsize=(16, 8))
    for mc in mcmcs:
        mle, mle_place, min_le = mle_values(mc, True)
        best_vals = mc['chain'][mle_place, :]
        print("{} - MLE: {:.2f} at {} with values: \n {}".format(mc['name'], mle, mle_place, best_vals), sep=",")
        my_alpha = 1 if mc['active'] else 0.4
        ax.plot(mc['ll'][mc['tally']:, 0], label=mc['name'], alpha = my_alpha, zorder=10*my_alpha)
        ax.scatter(mle_place, mle, s=300, zorder=10*my_alpha, edgecolor='k', linewidth=2, alpha = my_alpha)
    ax.legend()
    ax.set_title('Likelihood Progreesion')
    # ax.set_xlim((0,1000))
    return fig, ax
