# coding: utf-8

if __name__ == '__main__':

    from pertussis import *

    logger.setLevel(logging.INFO)
    # Initial
    r_start = 1929
    r_end = 2017
    step = 1 / N

    # Data
    data_M, data_M2, months, months2 = cases_monthly()
    # data_M = ma.array(data_M)
    # data_M[data_M > 150] = 150 #ma.masked
    state_0 = collect_state0()

    # # Make Model

    # ## Distributions

    dist_om = uniform(3.5, 1)
    dist_phi = uniform(0, 2 * np.pi)
    dist_rho = uniform(10, 50)
    dist_f = uniform(0.00, 0.5)
    dist_e = uniform(0.5, 0.5)
    dists = [dist_om, dist_phi, dist_rho, dist_f, dist_f, dist_f, dist_e]

    np.random.seed(314)
    # ## Build MCMC
    # Init or load

    # ### Load
    load = False
    # load = True
    # Load
    if load:
        how_many = 4
        vers = 'chains/1117-rho-60-multi-sigma-'
        vers = ''
        print('./' + vers)
        mcmcs = [load_mcmc('./' + vers + 'mcmc_v2_{}.pkl'.format(i)) for i in range(how_many)]
        print([len(mcmc['chain']) for mcmc in mcmcs])
        print([mcmc['max_likelihood'] for mcmc in mcmcs])
        new_path = vers
    ### Initialize

    else:

        # Initialize
        extra_params = dict(cov=2 * np.diag((0.2, np.pi / 10, 0.2, 5e-6, 5e-6, 5e-7)),
                            datay1=data_M,
                            datay2=data_M2,
                            datax1=months,
                            datax2=months2,
                            dists=dists,
                            names=['omega', 'phi', 'rho', 'f1', 'f2', 'f3'],
                            active_params=[1, 3, 4, 5],
                            # sigma=50)
                            sigma=np.array([50, 100, 100]).reshape(3, 1),
                            sigma2 = 100)
        mcmcs = []

        my_rho = 50
        print("V2")
        new_path = 'chains/{}/'.format(my_rho)
        try:
            os.mkdir(new_path)
        except FileExistsError:
            ('Overriding folder')
        # Chains
        om, phi, rho, f1, f2, f3, e = 3.9805, 1.45, my_rho, 0.0008, 0.0004, 0.00007, 1
        # om, phi, rho, f1, f2, f3, e = 3.9805, 3.1523154912649241, my_rho, 0.0009041826057846,0.0002610614564511,0.000071739716255, 1
        mcmc0 = init_mcmc('mcmc_0', state_0, r_start, r_end, om, phi, rho, f1, f2, f3,
                          **extra_params)
        mcmcs.append(mcmc0)

        om, phi, rho, f1, f2, f3, e = 3.9805, np.pi, my_rho, 0.0010, 0.00125, 0.00008, 1
        mcmc1 = init_mcmc('mcmc_1', state_0, r_start, r_end, om, phi, rho, f1, f2, f3,
                          **extra_params)
        mcmcs.append(mcmc1)

        # om, phi, rho, f1, f2, f3, e = 3.9805, 2.8256, my_rho, 0.0012, 0.0003, 0.0001, 1
        om, phi, rho, f1, f2, f3, e = 3.9805, 2.8256, my_rho, 0.0012, 0.0003, 0.0001, 1
        mcmc2 = init_mcmc('mcmc_2', state_0, r_start, r_end, om, phi, rho, f1, f2, f3,
                          **extra_params)
        mcmcs.append(mcmc2)

        # om, phi, rho, f1, f2, f3, e = 3.9805, np.pi / 2, my_rho, 0.009, 0.0005, 0.00006, 1
        om, phi, rho, f1, f2, f3, e = 3.9805, np.pi / 2, my_rho, 0.009, 0.0005, 0.00006, 1
        mcmc3 = init_mcmc('mcmc_3', state_0, r_start, r_end, om, phi, rho, f1, f2, f3,
                          **extra_params)
        mcmcs.append(mcmc3)

        mcmcs = sample(mcmcs, iterations=600, recalculate=150,
                       sd_stop_after=10000, scaling_stop_after=10000,
                       save_path='./'+new_path, do_gr=True)
        # exit()
    mcmcs = sample(mcmcs, iterations=15000, recalculate=250,
                   sd_stop_after=10000, scaling_stop_after=10000,
                   save_path='./'+new_path, do_gr=True)
