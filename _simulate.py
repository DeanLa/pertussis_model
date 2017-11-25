if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde as kde
    # from scipy.stats import norm, uniform, multivariate_normal as multinorm, norm
    # from tqdm import tqdm
    import seaborn as sns

    np.set_printoptions(edgeitems=10, linewidth=120, suppress=True, precision=8)

    from pertussis import *

    logger.setLevel(logging.INFO)

    # load = False
    load = True
    if load:
        # load simulation
        simulation = load_mcmc('./simulations/rho-60-36k.pkl')
        mcmc = simulation['mcmc']
        print(simulation['name'])
        print(len(simulation['p']))
        print(mcmc['name'])
        print(len(mcmc['chain']))
        print(mcmc['tally'])
    else:
        # Load MCMC
        mcmc = load_mcmc('./chains/1117-rho-60-multi-sigma-best-mcmc_v2_0.pkl')
        print(mcmc['name'], ': ', len(mcmc['chain']))
        print(mcmc['tally'])
        names = mcmc['names']
        # Take subsets
        take_subsets(mcmc)