import matplotlib.pyplot as plt
import pymc as pm
plt.style.use('ggplot')

def draw_no_split(x, y):
    # l = y.shape[1]
    l = len(y)
    fig, axs = plt.subplots(l, figsize=(12, 8))

    return fig, axs


def draw_split(x, y, which):
    l = len(y)
    fig = plt.figure(figsize=(14, 11))
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


def draw_model(x, y, labels=None, split=2, collapse=False):
    l = len(y)
    if l > 5:
        split = False
    if split:
        assert l <= 5, "Must have at most 5 variables"
        fig, axs = draw_split(x, y, which=split)
    else:
        fig, axs = draw_no_split(x, y)
    for i in range(l):
        try:
            ax = axs[i]
        except:
            ax=axs
        z = y[i][:,:]
        if collapse:
            z=z.sum(axis=0)
        lines = ax.plot(x, z.T, lw=1.3)
        try:
            ax.set_title(labels[i])
        except:
            ax.set_title("")
        ub = max(0.00001,z.max() * 1.05)
        # ub = 1
        ax.set_ylim([0,ub])
        # ax.set_xlim(left=1948)#, right = 2010)
        ax.plot([1957, 1957], [0, ub], "k--")
        ax.plot([2002, 2002], [0, ub], "k--")
        if l <= 3:
            ax.legend(lines, [j+1 for j in range(len(lines))], loc='lower left')
    if split:
        axs[split].legend(lines, [j+1 for j in range(len(lines))], loc='lower left')

    return fig, axs

# def draw_age_groups(x, y):
#     fig, axs = plt.subplots(4, 4)
#     axs = (np.hstack(axs))
#     prin

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