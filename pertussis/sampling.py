from pertussis import *
import threading
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal as multinorm
from time import sleep
from tqdm import tqdm  # , tqdm_notebook as tqnb


def sample_mcmc(mcmc, recalculate, sd_stop_after, scaling_stop_after):
    compute_scaling_factor = len(mcmc['chain']) < scaling_stop_after
    compute_new_sd = len(mcmc['chain']) < sd_stop_after
    dists = mcmc['dists']
    # print(mcmc['name'])
    for iteration in tqnb(range(recalculate), desc=mcmc['name'], leave=False, position=0):
        if not (mcmc['active']):  # When G-R converges, stop running all but one. -conition on GRB
            continue
        logger.info('*** CHAIN {} ITERATION {}'.format(mcmc['name'], len(mcmc['accepted'])))

        # Save Chain
        if len(mcmc['chain']) % 50 == 5: save_mcmc(mcmc)
        # Change parameters in warm up phase
        if iteration == recalculate - 1:
            # acceptance rate
            accept_star = np.mean(mcmc['accepted'][-recalculate:])
            mcmc['rates'] = np.append(mcmc['rates'], accept_star)
            # new scaling factor - pg. 24
            if compute_scaling_factor:
                new_scaling_factor = mcmc['scaling_factor'][-1]
                # new_scaling_factor *= np.e ** ((accept_star - mcmc['accept_hat']) / len(mcmc['scaling_factor']))
                new_scaling_factor *= np.e ** (accept_star - mcmc['accept_hat'])
                mcmc['scaling_factor'] = np.append(mcmc['scaling_factor'], new_scaling_factor)
            else:
                new_scaling_factor = 1
            # new cov pg. 24
            if compute_new_sd:
                new_cov_tmp = mcmc['cov'].copy()
                try:
                    sigma_star = np.cov(mcmc['chain'][-recalculate:, :].T)
                    new_cov = mcmc['cov'].copy() * 0.25 + 0.75 * sigma_star
                    proposed = multinorm(mcmc['values'], new_cov * new_scaling_factor ** 2)
                    mcmc['cov'] = new_cov
                except:
                    print("Singular COV at", len(mcmc['accepted']), mcmc['name'])
                    mcmc['cov'] = new_cov_tmp
            mcmc['sd'] = new_scaling_factor ** 2 * mcmc['cov']

        # Current Stats
        ll_now = log_liklihood(mcmc['y_now_M'], mcmc['datax'], mcmc['sigma'])

        # Pick new set
        try:
            proposed = multinorm(mcmc['values'], mcmc['sd'])  # Try new set ***************************************
        except Exception as e:
            print(e)
            print(mcmc['name'], "TUNRED OFF", len(mcmc['accepted']))
            mcmc['active'] = False
            continue
        guess = proposed.rvs()
        g = mcmc['initial_guess']
        g[mcmc['active_params']] = guess
        # Check if possible set
        g[1] %= 2 * np.pi  # Phi is special case #CHANGE THIS IF OMEGA NOT INCLUDED/NOT
        test = [dists[dist].pdf(guess[i]) <= 0 for i, dist in enumerate(mcmc['active_params'])]
        logger.info(str(g))
        if any(test):  # If test fails, LL is -inf
            logger.error('Bad prior at model {}\n{}\n{}'.format(mcmc['name'], guess, test))
            ll_star = -np.inf
            y_star_M, state_z = -np.inf * np.ones((3, 192)), -np.inf * np.ones(270)
        else:  # We can carry on with MCMC
            try:
                y_star_M, state_z = run_model(mcmc['state_0'], mcmc['start'], mcmc['end'], *g, e=1,
                                              r_0=40)  # RUN MODEL =============================================
                #                 logger.info(str(y_star_M))
                ll_star = log_liklihood(y_star_M, mcmc['datax'], mcmc['sigma'])
                logger.info(str(ll_star))
                if ll_star < -99999:  # Something bad happend BY DESIGN at model
                    logger.warning('bad set for model {} with guess {}'.format(mcmc['name'], iteration))
            except:
                logger.error('exception at model {} PROBABLY S-I-R fail'.format(mcmc['name']))
                ll_star = -np.inf

        # when possible, run model
        log_r = ll_star - ll_now
        draw = np.random.rand()

        if log_r > np.log(draw):
            mcmc['values'] = guess.copy()
            mcmc['accepted'] = np.append(mcmc['accepted'], 1)
            mcmc['y_now_M'] = y_star_M
        else:
            mcmc['accepted'] = np.append(mcmc['accepted'], 0)

        # Update
        mcmc['chain'] = np.vstack((mcmc['chain'], mcmc['values']))
        mcmc['guesses'] = np.vstack((mcmc['guesses'], guess))
        mcmc['y_hat_M'] = np.concatenate((mcmc['y_hat_M'], y_star_M[None, :, :]), axis=0)
        mcmc['max_likelihood'] = np.max((mcmc['max_likelihood'], ll_now))
        mcmc['ll'] = np.vstack((mcmc['ll'], np.array([ll_now, ll_star])))
        mcmc['state_z'] = np.vstack((mcmc['state_z'], state_z))


def sample_multi(mcmcs, recalculate, sd_stop_after, scaling_stop_after):
    threads = []
    for mcmc in mcmcs:
        t = threading.Thread(target=sample_mcmc, args=(mcmc, recalculate, sd_stop_after, scaling_stop_after))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


def sample(mcmcs, iterations=1000, recalculate=250, sd_stop_after=5000, scaling_stop_after=5000):
    # mcmcs[2]['active'] = False

    iter_over = np.ones(iterations // recalculate) * recalculate
    iter_mod = iterations % recalculate
    if iter_mod: iter_over = np.append(iter_over, iter_mod)
    print(iter_over)
    for mini in tqnb((iter_over)):
        sample_multi(mcmcs, recalculate, sd_stop_after, scaling_stop_after)
        if sum([mc['active'] for mc in mcmcs]) > 1:
            gelman_rubin_test(mcmcs)


def gelman_rubin_test(mcmcs):
    gr_curr = gelman_rubin(make_chains([mc for mc in mcmcs if mc['active']]))
    mcmcs[0]['gelman_rubin'] = np.vstack((mcmcs[0]['gelman_rubin'], gr_curr))
    if len(mcmcs[0]['chain']) > 2000:
        for gri in range(len(mcmcs) - 1):
            for grj in range(gri + 1, len(mcmcs)):
                if (mcmcs[gri]['active']) and (mcmcs[grj]['active']):
                    grtmp = gelman_rubin(make_chains([mcmcs[gri], mcmcs[grj]]))
                    if all(grtmp < 1.1):
                        mcmcs[grj]['active'] = False
                        save_mcmc(mcmcs[grj])
                        print("Shutting off {} by {}".format(mcmcs[grj]['name'], mcmcs[gri]['name']))
                        print(grtmp)
