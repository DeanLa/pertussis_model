from pertussis import *

from pprint import pprint
from time import clock, sleep
import numpy as np
import scipy.stats as stats
from tqdm import tqdm, tqdm_notebook as tqnb


def reduce_month(y, r, to_yearly=False):
    if to_yearly == True:
        r *= 12
    # print (y.shape)
    new_y = np.split(y, np.cumsum(sc)[:-1])
    # print (new_y[0].shape)
    # print (new_y[1].shape)
    new_y = [yi.sum(axis=0) for yi in new_y]
    # print (new_y[0].shape)
    # print (new_y[1].shape)
    new_y = [np.sum(yi.reshape(-1, r), axis=1) for yi in new_y]
    # print (new_y[0].shape)
    # print (new_y[1].shape)
    new_y = np.array(new_y)
    # print (new_y.shape)
    return new_y


def init_simulation(name, mcmc, policies=(), **kwargs):
    import copy
    extra = copy.deepcopy(kwargs)
    sim = {'name': name}
    sim['mcmc'] = mcmc
    sim['policies'] = policies
    sim['n_policies'] = len(policies)
    # Info
    sim['start'] = mcmc['end']
    sim['end'] = mcmc['end'] + extra.get('simulation_length', 12)
    # Hospital dist (From Excel Data) - DO not change
    sim['data_sick'] = np.array([1647, 9761, 6832])  # * p
    sim['data_hospital'] = np.array([557, 201, 129])
    a = 1 + sim['data_hospital']
    b = 1 + sim['data_sick'] - sim['data_hospital']
    sim['dist'] = stats.beta(a, b)

    sim['subset_pick'] = []
    sim['p'] = []
    sim['pregnant_coverage'] = []
    # Simulation Traces
    # sim_length = 10000
    # l = len(mcmc['chain_subset'])
    # sim['subset_pick'] = np.random.randint(l,size=sim_length)
    # sim['p'] = dist.rvs(size=(sim_length,3))
    # sim['pregnant_coverage'] = stats.uniform(0, 1).rvs(lsim_length)

    sim = {**sim, **extra}
    return sim


def init_policy(name, **kwargs):
    import copy
    extra = copy.deepcopy(kwargs)
    policy = {'name': name}
    policy['history'] = "mcmc"
    # Meta
    # policy['era'] = 10 # Years
    # Demographics
    policy['delta'] = delta[-1]
    policy['mu'] = mu[-1, :]
    policy['aliyah'] = aliyah[-1, :]

    coverage = 0.95
    # Vaccines
    if 'vax_ages' in extra.keys():
        _vax_ages = extra['vax_ages']
        assert all(np.in1d(_vax_ages, a_l)), "Vaccine should happen on listed Age Group:\n {}".format(_ages)
        _vax_ap = np.in1d(a_l, _vax_ages).astype(int)  # Currently holds indexes for efficacy
        _vax_ap = coverage * _vax_ap.astype(float)[:-1]
    else:
        _vax_ap = vax_ap
    if 'dynamic' in extra.keys():
        _dynamic_ages = extra['dynamic']
        assert all(np.in1d(_dynamic_ages, a_u)), "Dynamic Vaccine should happen on listed Age Group:\n {}".format(_ages)
        _dynamic_ap = np.in1d(a_u, _dynamic_ages).astype(int)
        _dynamic_ap = coverage * _dynamic_ap.astype(float)[:-1]
        policy['dynamic_ap'] = _dynamic_ap
    if 'control' in extra.keys():
        _control_ages = extra['control']
        assert all(np.in1d(_control_ages, a_u)), "Control Vaccine should happen on listed Age Group:\n {}".format(_ages)
        _control_ap = np.in1d(a_u, _control_ages).astype(int)
        _control_ap = 0.25 * coverage * _control_ap.astype(float)[:-1]
        policy['control_ap'] = _control_ap
    # policy['pregnant_coverage'] = 0.95
    policy['vax_ap'] = _vax_ap

    # Chains and results
    policy['vaccines'] = []
    policy['sick'] = []
    policy['hospital'] = []
    # policy['p'] = []
    # policy['picks'] = []
    policy['done'] = 0
    policy = {**policy, **extra}
    return policy


def predict_model(state_0, start, end,
                  rho, om, phi, f, e=1,
                  r=3, policy=None):
    # pprint (policy)
    '''Runs the model for future years'''
    # logger.setLevel(logging.DEBUG)  # To capture first warning
    r = 1 / r
    # print (start, end)
    timeline = np.arange(start, end, N * r)
    num_months = len(timeline)
    no_likelihood = ""
    e_ap = e  # Helper - vaccine unefficacy
    e_wp = (1 - epsilon_wp) / alpha_wp  # Helper - vaccine unefficacy
    alpha_ap = (1 - epsilon_ap) / e_ap
    # print ("alpha_ap====================================================")
    # print (alpha_ap)
    # Initiate return matrices and start with values from state_0
    S, Vap, Vwp, Is, Ia, R, Healthy, All, Shots, New = [np.zeros((J, num_months)) for _ in range(10)]  # 9 Matrices
    for i, c in enumerate([S, Vap, Vwp, Is, Ia, R]):
        c[:, 0] = state_0[i]
        All[:, 0] += c[:, 0]
    New[:, 0] = state_0[9]
    # ************************************************* Start Loop ***************************************
    # print (timeline[[1,-1]])
    for t, T in enumerate(timeline[1:], start=1):
        # print (T)
        # exit()
        # T: real time, t: days since start
        demographic_change = 2300 >= T >= data_start

        #        t_max = min(2014 - data_start, #75
        #                    max(0, int(T) - data_start))  # 2014 where it ends
        #        t_max = 2014 - data_start
        # Compartments and Derivatives
        A = All[:, t - 1].sum()
        if A > 35e6:
            logger.setLevel(logging.INFO)
            logger.error("Too much people at time: {:.2f}".format(T))
            return no_likelihood
        # nS, nVap, nVwp, nIs, nIa, nR = [c[:, t - 1] / A for c in [S, Vap, Vwp, Is, Ia, R]]
        nS = np.maximum(0, S[:, t - 1] / A)
        nVap = np.maximum(0, Vap[:, t - 1] / A)
        nVwp = np.maximum(0, Vwp[:, t - 1] / A)
        nIs = np.maximum(0, Is[:, t - 1] / A)
        nIa = np.maximum(0, Ia[:, t - 1] / A)
        nR = np.maximum(0, R[:, t - 1] / A)
        nA = nS + nVap + nVwp + nIs + nIa + nR
        # Initialize return values
        for i, c in enumerate([S, Vap, Vwp, Is, Ia, R]):
            c[:, t] = c[:, t - 1] / A

        ## Helpers
        nI = nIa + nIs
        beta_ = 0 + f * beta(T, om, phi, rho)
        beta_ = np.where(beta_ >= 1e-9, beta_, 1e-9)  # Take only non-negative valuese
        IC = nI.dot(C)
        lambda_ = beta_ * IC  # Needs to be normalized
        if any(lambda_ < 0):
            # "{}\nlambda {}\n>0 {}\nbeta {}\n IC {} \n I {} \n\n".format([T, t], lambda_, lambda_ >= -0., beta_,
            #                                                             state_0, nI))
            logger.warning('0 likelihood')
            logger.error('0 likelihood lambda<0')
            logger.error("rho {}\nomega {}\nphi {}\nf {} \n\n".format(rho, om, phi, f))
            logger.setLevel(logging.INFO)
            return no_likelihood

        ##### Equations ####
        ## Susceptible
        # Birth
        if demographic_change:
            babies = policy['delta'] * r
            preg_cover = policy['pregnant_coverage']
            Vap[0, t] += babies * preg_cover
            S[0, t] += babies * (1 - preg_cover)  # IN: Birth rate

        # Transmission
        S[:, t] += r * (omega * nR + omega_ap * nVap + omega_wp * nVwp)  # IN: Waning from Natural and vaccine
        S[:, t] -= r * lambda_ * nS  # OUT: Becoming infected
        if any(S[:, t] < 0):
            logger.warning("S < 0 at {} \n {}".format(T, S[:, t]))
            logger.setLevel(logging.ERROR)
        ##### Vaccination
        vax = policy['vax_ap'].copy()

        if 'dynamic' in policy.keys():
            if int(T) % 4 == policy['mod_year']:
                # ADD TO VAX AGES
                vax += policy['dynamic_ap']
        if 'control' in policy.keys():
            # ADD TO VAX AGES
            vax += policy['control_ap']
        # print (vax)tak

        Vap[1:, t] += r * vax * a * nS[:-1]  # IN: Age and vaccinate aP from S classes
        S[1:, t] += r * (1 - vax) * a * nS[:-1]  # IN: Age from previous age no vaccine

        Shots[1:, t] += r * vax * a * nA[:-1]  ### COUNTER

        Vap[:, t] -= r * lambda_ * e_ap * nVap  # OUT: Getting sick (reduced by efficacy)
        Vwp[:, t] -= r * lambda_ * e_wp * nVwp  # OUT: Getting sick (reduced by efficacy)
        Vap[:, t] -= r * omega_ap * nVap  # OUT: Waning to S
        Vwp[:, t] -= r * omega_wp * nVwp  # OUT: Waning

        # Infected
        I_ap = r * lambda_ * e_ap * nVap  # HELPER: Infected with ap
        I_wp = r * lambda_ * (e_wp * nVwp + nS)  # HELPER: Infected with wp or no vaccine

        New[:, t] = alpha_ap * I_ap + alpha_wp * I_wp  # New infected - this is the most important for fit
        # if t==1:
        #     print (alpha_ap)
        #     print (alpha_wp)
        Is[:, t] += alpha_ap * I_ap + alpha_wp * I_wp  # IN: Infected with symptoms chance
        Is[:, t] -= r * gamma_s * nIs  # OUT: Recovered

        Ia[:, t] += (1 - alpha_ap) * I_ap + (1 - alpha_wp) * I_wp  # IN: Infected with NO symptoms chance
        Ia[:, t] -= r * gamma_a * nIa  # OUT: Recovered

        # Recovered
        R[:, t] += r * (gamma_s * nIs + gamma_a * nIa)  # IN: Recovered from I
        R[:, t] -= r * omega * nR  # OUT: Natrual waning

        ## Regular Aging
        if True:
            # S
            S[:-1, t] -= r * a * nS[:-1]  # OUT: Age out to Next V or Next S, OUT is the same
            # V
            # a_corr_ap = age_correction(2002, T, a_u)  # get the age transition correction vector for ap
            a_corr_ap = 1  # get the age transition correction vector for ap
            a_corr_wp = age_correction(1957, T, a_u)  # get the age transition correction vector for wp
            Vap[1:, t] += r * a * nVap[:-1] * a_corr_ap  # IN
            Vap[:-1, t] -= r * a * nVap[:-1] * a_corr_ap  # OUT - Next Age Group
            Vwp[1:, t] += r * a * nVwp[:-1] * a_corr_wp  # IN
            Vwp[:-1, t] -= r * a * nVwp[:-1] * a_corr_wp  # OUT
            # print (a.shape,  Is[:-1].shape, nVwp[:-1].shape)
            # I and R
            Is[1:, t] += r * a * nIs[:-1]  # IN
            Is[:-1, t] -= r * a * nIs[:-1]  # OUT
            Ia[1:, t] += r * a * nIa[:-1]  # IN
            Ia[:-1, t] -= r * a * nIa[:-1]  # OUT
            R[1:, t] += r * a * nR[:-1]  # IN
            R[:-1, t] -= r * a * nR[:-1]  # OUT

        # Kill and Update A
        mu_current = r * policy['mu']
        aliyah_current = r * policy['aliyah']
        for i, c in enumerate([S, Vap, Vwp, Is, Ia, R]):
            add_aliyah = (c[:, t - 1] / All[:, t - 1]) * aliyah_current
            c[:, t] = c[:, t] * A
            if demographic_change:
                c[:, t] -= mu_current * c[:, t - 1]
                c[:, t] += add_aliyah
            All[:, t] += c[:, t]
        Shots[:, t] *= A
        New[:, t] *= A
    Healthy = S + Vap + Vwp
    logger.setLevel(logging.INFO)
    # print (New)
    # print(New[:, 0].shape)
    # print(New[:, 0])
    return [S, Vap, Vwp, Is, Ia, R, Healthy, All, Shots, New]


def simulate_future(simulation, iterations=1000, r=3):
    policies = simulation['policies']
    mcmc = simulation['mcmc']
    start = simulation['start']
    end = simulation['end']

    # Get relevant subsets
    try:
        subset_chain = mcmc['chain_subset']
        subset_states = mcmc['state_z_subset']
        l = len(subset_chain)
    except:
        logger.error("No chain subset in MCMC")
        print("No chain subset in MCMC")
        exit()
    print("L", l)

    # Simulate futures
    for i in tqnb(range(iterations)):
        if i % 20 == 1:
            save_mcmc(simulation, './simulations/')
        # pick number then take from same place and sample from dists
        pick = np.random.randint(0, l)
        beta_p = simulation['dist'].rvs()
        # uniform_preg_cov = stats.uniform(0.25, 0.75).rvs()
        uniform_preg_cov = stats.uniform(0, 1).rvs()
        # print(uniform_preg_cov)

        # save chains
        simulation['subset_pick'].append(pick)
        simulation['p'].append(beta_p)
        simulation['pregnant_coverage'].append(uniform_preg_cov)
        # Pick set and state_z from mcmc
        state_0 = subset_states[pick, :]
        state_0 = np.split(state_0, 10)
        params = mcmc['initial_guess'].copy()
        params[mcmc['active_params']] = subset_chain[pick, :]
        om, phi, rho, f1, f2, f3 = params
        f = np.concatenate((nums(f1, sc[0]), nums(f2, sc[1]), nums(f3, sc[2])))

        # Run on all poclies
        for policy in tqnb(policies, leave=False):
            policy['pregnant_coverage'] = uniform_preg_cov
            # print (policy['pregnant_coverage'])
            # print ( policy['vax_ap'])
            policy['vax_ap'][21] = policy['pregnant_coverage'] * 0.5  # Vaccinate pregnant
            y = predict_model(state_0, start, end,
                              rho, om, phi, f, e=1,
                              r=r, policy=policy)
            # break
            # policy['results'] = y
            sick = reduce_month(y[-1], r) * p # Outpatients
            vaccines = reduce_month(y[-2], r)
            policy['sick'].append(sick.sum(axis=1))
            policy['hospital'].append(sick.sum(axis=1) * beta_p)
            policy['vaccines'].append(vaccines.sum(axis=1))
            # break
    save_mcmc(simulation, './simulations/')


def predict_soon(simulation, r=3):
    policy = simulation['policies'][0].copy()
    mcmc = simulation['mcmc']
    start = simulation['start']
    more = 8
    end = simulation['start'] + more

    # Get relevant subsets
    try:
        subset_chain = mcmc['chain_subset']
        subset_states = mcmc['state_z_subset']
        l = len(subset_chain)
    except:
        print("No chain subset in MCMC")
        exit()
    print("L", l)
    mcmc['prediction'] = np.zeros((0, 3, more * 12))
    for pick in tqnb(range(l)):
        # Pick set and state_z from mcmc
        state_0 = subset_states[pick, :]
        state_0 = np.split(state_0, 10)
        params = mcmc['initial_guess'].copy()
        params[mcmc['active_params']] = subset_chain[pick, :]
        om, phi, rho, f1, f2, f3 = params
        f = np.concatenate((nums(f1, sc[0]), nums(f2, sc[1]), nums(f3, sc[2])))
        # policy['pregnant_coverage'] = 0
        y = predict_model(state_0, start, end,
                          rho, om, phi, f, e=1,
                          r=r, policy=policy)
        # print ('---------------------------')
        # print(y[-1].shape)
        # print(y[-1][:,0])
        # y = difference_model(state_0, start, end,
        #                      rho, om, phi, f, 1,
        #                      r=r, full_output=True)
        # y, _ = run_model(state_0, 1998, 2014,
        #                  om, phi, rho, f1,f2,f3,e=1,r=3, r_0=20,years_prior=1)

        # break
        # policy['results'] = y
        # sick = reduce_month(y[-1]/y[-3].sum(axis=0), r)
        # all = y[-3]

        A = y[-3]
        # print (A[:,0])
        # A = 1
        y = y[-1]
        # print (y[:,0])
        # # Take result and sum values according to susceptibility
        y = np.split(y, np.cumsum(sc)[:-1])
        # print(y[1].shape)
        y = [yi.sum(axis=0) for yi in y]
        # print(y[1].shape)
        # print (y)
        # print (A.sum(axis=0))
        y = [yi / A.sum(axis=0) for yi in y]
        # print (y[1].shape)
        # print (y[1])
        y = [np.sum(yi.reshape(-1, r), axis=1) for yi in y]
        # print (y[1].shape)
        # print (y[1])
        # print (y)
        y = np.array(y)
        per100k = y * 10 ** 5  # y[:, -48:]
        # print (per100k.shape)
        mcmc['prediction'] = np.concatenate((mcmc['prediction'], per100k[None, :, :]), axis=0)
        # break


def compare_policies(simulation):
    pass


def take_subsets(mcmc):
    tally = mcmc['tally']
    chain = mcmc['chain'][tally:, :]
    # Pick Chains for Simulation
    l = len(chain)
    chain_ess = ess(mcmc)
    print('Effective sample size: {}'.format(chain_ess))
    thinning = (l // chain_ess).max().astype(int)
    # thinning = 9000
    sl = np.arange(tally, len(mcmc['chain']), thinning)
    # print(sl)
    # mcmc['picks'] = sl
    mcmc['state_z_subset'] = mcmc['state_z'][sl, :]
    mcmc['chain_subset'] = mcmc['chain'][sl, :]
    cut = mcmc['state_z_subset'].sum(axis=1) > 0
    print(cut.shape)
    mcmc['chain_subset'] = mcmc['chain_subset'][cut, :]
    mcmc['state_z_subset'] = mcmc['state_z_subset'][cut, :]
    mcmc['index_subset'] = sl[cut]
    print('Subset length: {}'.format(len(mcmc['chain_subset'])))
    #

def test():
    policy1 = init_policy('test_default')
    policy2 = init_policy('test_max', vax_ages=a_u)
    mcmc = {'chain_subset': np.arange(700).reshape(100, 7),
            'state_z_subset': [i * np.ones(10).reshape(5, 2) for i in range(100)]}
    mcmc = load_mcmc('./chains/03027-23k-multimcmc_0.pkl')
    pprint(policy2['vax_ap'])
    pprint(policy1['vax_ap'].shape)

    exit()
    sim = init_simulation('test_sim', mcmc)
    sim['policies'] = [policy1, policy2]
    simulate_future(sim, iterations=3)
    save_mcmc(sim, path='./policies/')
    exit()
    print(policy1['picks'])
    print(policy2['picks'])


def create_pairwise(simulation):
    policies = simulation['policies']
    default = policies[0]
    metric_names = ['sick', 'hospital', 'vaccines']
    base = {}
    for m in metric_names:
        base[m] = np.array(default[m])
    # Make differences from default
    for policy in simulation['policies']:
        for m in metric_names:
            policy[m+'_diff'] = base[m] - np.array(policy[m])
            policy[m+'_pct'] = 100 * policy[m+'_diff'].sum(axis=1) / base[m].sum(axis=1)
        # policy['hospital_diff'] = np.array(default['hospital']) - np.array(policy['hospital'])
        # policy['vaccines_diff'] = np.array(default['vaccines']) - np.array(policy['vaccines'])
        # policy['ratio'] = (np.array(policy['sick'])).sum(axis=1) / (np.array(policy['vaccines'])).sum(axis=1)
        # IMPORTANT: Default has to be first!!!!!
        #     policy['ratio_diff'] = policy['ratio'] - default['ratio']
        # policy['ratio_diff'] = np.array(policy['sick_diff']).sum(axis=1) / np.array(policy['vaccines']).sum(axis=1)

def simulate_future_mp(simulation, iterations=1000, r=3):
    # sim['p'] = []
    # sim['subset_pick'] = []
    # sim['pregnant_coverage'] = []
    # sim['done'] = 0
    policies = simulation['policies']
    mcmc = simulation['mcmc']
    start = simulation['start']
    end = simulation['end']

    # Get relevant subsets
    subset_chain = mcmc['chain_subset']
    subset_states = mcmc['state_z_subset']
    l = len(subset_chain)
    print("L", l)
    # Simulate futures

def simulate_policy(policy, simulation):
    mcmc = simulation['mcmc']
    start = simulation['start']
    end = simulation['end']
    subset_chain = mcmc['chain_subset']
    subset_states = mcmc['state_z_subset']
    l = len(subset_chain)
    for i in tqnb(range(policy['done'],iterations)):

        # for policy in tqnb(policies, leave=False):
        # pick number then take from same place and sample from dists
        pick = sim['subset_pick'][i]
        beta_p = sim['p'][i]
        uniform_preg_cov = sim['pregnant_coverage'][i]
        # Pick set and state_z from mcmc
        state_0 = subset_states[pick, :]
        state_0 = np.split(state_0, 10)
        params = mcmc['initial_guess'].copy()
        params[mcmc['active_params']] = subset_chain[pick, :]
        om, phi, rho, f1, f2, f3 = params
        f = np.concatenate((nums(f1, sc[0]), nums(f2, sc[1]), nums(f3, sc[2])))

        policy['vax_ap'][21] = uniform_preg_cov * 0.5  # Vaccinate pregnant
        y = predict_model(state_0, start, end,
                          rho, om, phi, f, e=1,
                          r=r, policy=policy)
        # break
        # policy['results'] = y
        sick = reduce_month(y[-1], r) * p # Outpatients
        vaccines = reduce_month(y[-2], r)
        policy['sick'].append(sick.sum(axis=1))
        policy['hospital'].append(sick.sum(axis=1) * beta_p)
        policy['vaccines'].append(vaccines.sum(axis=1))
        # break
        mcmc['done'] += 1
    return policy
