import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# from seaborn import color_palette

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
