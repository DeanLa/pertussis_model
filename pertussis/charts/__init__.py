import matplotlib.pyplot as plt
from .mcmc import *
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
        ax = axs[i]
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