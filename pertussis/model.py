from pertussis import *

from pprint import pprint
from time import clock, sleep
from scipy.integrate import odeint
from copy import copy
import numpy as np


def difference_model(state_0, start, end,
                     rho, om, phi, f, e,
                     r=1, full_output=False):
    '''r is a hyper parameter that corresponds with resolution'''
    # num_months *= r
    # logger.setLevel(logging.DEBUG)  # To capture first warning
    r = 1 / r
    # print (start, end)
    timeline = np.arange(start, end, N * r)
    num_months = len(timeline)
    # no_likelihood = -np.inf * np.ones((J, num_months))  # Vector to return if run is not good
    no_likelihood = ""
    e_ap = e  # Helper - vaccine unefficacy
    e_wp = (1 - epsilon_wp) / alpha_wp  # Helper - vaccine unefficacy
    alpha_ap = (1 - epsilon_ap) / e_ap

    # Initiate return matrices and start with values from state_0
    S, Vap, Vwp, Is, Ia, R, Healthy, All, New = [np.zeros((J, num_months)) for _ in range(9)]  # 9 Matrices
    for i, c in enumerate([S, Vap, Vwp, Is, Ia, R]):
        c[:, 0] = state_0[i]
        All[:, 0] += c[:, 0]
    #************************************************* Start Loop ***************************************
    for t, T in enumerate(timeline[1:], start=1):
        demographic_change = 2300 >= T >= data_start

        t_max = min(2014 - data_start,
                    max(0, int(T) - data_start))  # 2014 where it ends
        # Compartments and Derivatives
        A = All[:, t - 1].sum()
        if A > 9e6:
            logger.setLevel(logging.INFO)
            logger.error("Too much people at time: {:.2f}".format(T))
            # return [S, Vap, Vwp, Is, Ia, R, Healthy, All, New]
            return no_likelihood
        # nS, nVap, nVwp, nIs, nIa, nR = [c[:, t - 1] / A for c in [S, Vap, Vwp, Is, Ia, R]]
        nS = np.maximum(1e-9, S[:, t - 1] / A)
        nVap = np.maximum(1e-9, Vap[:, t - 1] / A)
        nVwp = np.maximum(1e-9, Vwp[:, t - 1] / A)
        nIs = np.maximum(1e-9, Is[:, t - 1] / A)
        nIa = np.maximum(1e-9, Ia[:, t - 1] / A)
        nR = np.maximum(1e-9, R[:, t - 1] / A)
        # Initialize return values
        for i, c in enumerate([S, Vap, Vwp, Is, Ia, R]):
            c[:, t] = c[:, t - 1] / A

        ## Helpers
        nI = nIa + nIs
        beta_ = 0 + f * beta(T, om, phi, rho)
        beta_ = np.where(beta_ >=1e-9, beta_, 1e-9)  # Take only non-negative valuese
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
            S[0, t] += delta[t_max] * r  # IN: Birth rate
        # Transmission
        S[:, t] += r * (omega * nR + omega_ap * nVap + omega_wp * nVwp)  # IN: Waning from Natural and vaccine
        S[:, t] -= r * lambda_ * nS  # OUT: Becoming infected
        if any(S[:, t] < 0):
            logger.warning("S < 0 at {} \n {}".format(T, S[:, t]))
            logger.setLevel(logging.ERROR)
        ##### Vaccination
        if data_start <= T < 1957:
            S[1:, t] += r * a * nS[:-1]  # IN
        if 1957 <= T < 2002:  # Begin wp
            Vwp[1:, t] += r * vax_wp * a * nS[:-1]  # IN: Age and vaccinate wP from S classes
            S[1:, t] += r * (1 - vax_wp) * a * nS[:-1]  # IN: Age from previous age no vaccine
        if T >= 2002:  # Begin aP
            Vap[1:, t] += r * vax_ap * a * nS[:-1]  # IN: Age and vaccinate aP from S classes
            S[1:, t] += r * (1 - vax_ap) * a * nS[:-1]  # IN: Age from previous age no vaccine

        Vap[:, t] -= r * lambda_ * e_ap * nVap  # OUT: Getting sick (reduced by efficacy)
        Vwp[:, t] -= r * lambda_ * e_wp * nVwp  # OUT: Getting sick (reduced by efficacy)
        Vap[:, t] -= r * omega_ap * nVap  # OUT: Waning to S
        Vwp[:, t] -= r * omega_wp * nVwp  # OUT: Waning

        # Infected
        I_ap = r * lambda_ * e_ap * nVap  # HELPER: Infected with ap
        I_wp = r * lambda_ * (e_wp * nVwp + nS)  # HELPER: Infected with wp or no vaccine

        New[:, t] = alpha_ap * I_ap + alpha_wp * I_wp  # New infected - this is the most important for fit

        Is[:, t] += alpha_ap * I_ap + alpha_wp * I_wp  # IN: Infected with symptoms chance
        Is[:, t] -= r * gamma_s * nIs  # OUT: Recovered
        # Is[:, t] += M

        Ia[:, t] += (1 - alpha_ap) * I_ap + (1 - alpha_wp) * I_wp  # IN: Infected with NO symptoms chance
        Ia[:, t] -= r * gamma_a * nIa  # OUT: Recovered
        # Ia[:, t] += M

        # Recovered
        R[:, t] += r * (gamma_s * nIs + gamma_a * nIa)  # IN: Recovered from I
        R[:, t] -= r * omega * nR  # OUT: Natrual waning
        # R[:, t] -= 2 * M

        ## Regular Aging
        if demographic_change:
            # S
            S[:-1, t] -= r * a * nS[:-1]  # OUT: Age out to Next V or Next S, OUT is the same
            # V
            a_corr_ap = age_correction(2002, T, a_u)  # get the age transition correction vector for ap
            a_corr_wp = age_correction(1957, T, a_u)  # get the age transition correction vector for wp
            Vap[1:, t] += r * a * nVap[:-1] * a_corr_ap  # IN
            Vap[:-1, t] -= r * a * nVap[:-1] * a_corr_ap  # OUT
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
        mu_current = r * mu[t_max]
        aliyah_current = r * aliyah[t_max]
        for i, c in enumerate([S, Vap, Vwp, Is, Ia, R]):
            add_aliyah = (c[:, t - 1] / All[:, t - 1]) * aliyah_current
            c[:, t] = c[:, t] * A
            if demographic_change:
                c[:, t] -= mu_current * c[:, t - 1]
                c[:, t] += add_aliyah
            All[:, t] += c[:, t]
        New[:, t] *= A
    Healthy = S + Vap + Vwp
    logger.setLevel(logging.INFO)
    if full_output:
        return [S, Vap, Vwp, Is, Ia, R, Healthy, All, New]
    # print (New.shape)
    return New


def run_model(state_0, start, end,
              om, phi, rho, f1, f2, f3, e,
              r=3, r_0=20, years_prior=10):
    no_likelihood = -np.inf * np.ones((3,192))
    # fix and set params
    f = np.concatenate((nums(f1, sc[0]), nums(f2, sc[1]), nums(f3, sc[2])))
    phi = phi % (2 * np.pi)
    # Run first few years on high resolution
    y_0 = difference_model(state_0, start - years_prior, start,
                           rho, om, phi, f, e,
                           r=r_0, full_output=True)
    if type(y_0) == str:
        logger.error('Model exited at WARM UP')
        return no_likelihood
    # Take Main state_0 as last state of initial years
    state_0 = [yi[:, -1] for yi in y_0]
    # Run model
    y = difference_model(state_0, start, end,
                         rho, om, phi, f, e,
                         r=r, full_output=True)
    if type(y) == str:
        logger.error('Model exited at MAIN')
        return no_likelihood
    # If under 5 are not very 4/6 in the R comp. return no likelihood
    # Compute where the cut is
    tmp_cut = np.arange(int(r/N * (1953 - start)),
                        int(r / N * (1957 - start)))
    # Compute Total in the point, and R in the point
    R, A = y[5], y[7]
    tmp_R = R[9,tmp_cut]
    tmp_A = A[9,tmp_cut]
    mean_sick = (tmp_R / tmp_A).mean()
    # exit()
    # Test condition
    if  mean_sick<= 0.7:
        logger.error('Not enough sick kids, {:.2f}'.format(mean_sick))
        return no_likelihood
    y = y[-1]
    # Take result and sum values according to susceptibility
    y = np.split(y, np.cumsum(sc)[:-1])
    y = [yi.sum(axis=0)/A.sum(axis=0) for yi in y]
    # print((y[1]/A.sum(axis=0)).shape)
    y = [np.sum(yi.reshape(-1, r), axis=1) for yi in y]
    y = np.array(y) * 10**5
    # y is only new sick people on 3 age (susceptibiliy change) groups: 3,192

    # Monthly
    # Slices relevant for months data
    start_ix = (1998 - start) * 12
    end_ix = (2014 - start) * 12
    monthly = y[:, start_ix:end_ix]

    # Yearly
    # start_ix = 1951 - start
    # end_ix = 1999 - start
    # yearly = y.sum(axis=0)
    # yearly = np.sum(yearly.reshape(-1, 12), axis=1)
    # yearly = yearly[start_ix:end_ix]
    # return (monthly, yearly)  # This is 3 lists of 192 Monthly points and 48 yearly
    return monthly


def init_mcmc(name, state_0, r_start, r_end, *params, **kwargs):
    mcmc = {'name': name}  # Name- also name of file

    # Constatns
    mcmc['accept_hat'] = 0.23
    mcmc['sigma'] = 100
    mcmc['state_0'] = state_0
    mcmc['start'] = r_start
    mcmc['end'] = r_end
    mcmc['names'] = ['omega','phi','rho','f1','f2','f3','e']

    # Initial Values
    mcmc['values'] = np.array(params)  # CURRENT values ########################################
    M_now = run_model(state_0, r_start, r_end, *mcmc['values'])
    mcmc['y_now_M'] = M_now[None, :, :]  # CURRENT values ########################################
    # mcmc['y_now_Y'] = Y_now.copy()  # CURRENT values ########################################
    mcmc['active'] = True

    # Model Soecific
    mcmc['d'] = len(mcmc['values'])
    mcmc['cov'] = np.diag([0.1, np.pi / 10, 0.2, 0.04, 0.0004, 0.0004, 0.05])
    # mcmc['cov'] /= 100
    mcmc['scaling_factor'] = np.array([2.4 / np.sqrt(mcmc['d'])])
    mcmc['sd'] = mcmc['scaling_factor'] ** 2 * mcmc['cov']

    # Chains and Metrics
    mcmc['y_hat_M'] = mcmc['y_now_M'].copy()  # CHAIN: y values for proposed set (shape like data points)
    # mcmc['y_hat_Y'] = mcmc['y_now_Y'].copy()  # CHAIN: y values for proposed set (shape like data points)
    mcmc['chain'] = mcmc['values'].copy()  # CHAIN: proposed set
    mcmc['guesses'] = mcmc['values'].copy()  # CHAIN: proposed set
    mcmc['ll'] = np.array([-np.inf, -np.inf])
    mcmc['accepted'] = np.array([1])
    mcmc['rates'] = np.zeros(mcmc['d'])
    mcmc['max_likelihood'] = -np.inf
    mcmc['gelman_rubin'] = np.zeros(mcmc['d'])
    mcmc['change'] = np.array([-1])


    mcmc = {**mcmc, **kwargs.copy()}
    return mcmc


def make_model(data1, data2, state_0, r_start, r_end):
    '''data 1: monthely on 3 groups
    '''
    import pymc as pm
    om = pm.Normal('om', 4, 1 / 0.5 ** 2, value=4)
    # om.use_step_method(pm.Metropolis, sd_proposal=0.05)
    phi = pm.Uniform('phi', -np.pi, np.pi, value=0)
    f_top = 0.1
    f1 = pm.Uniform('f1', 0.001, f_top, value=0.05)
    f2 = pm.Uniform('f2', 0.001, f_top, value=0.05)
    f3 = pm.Uniform('f3', 0.001, f_top, value=0.05)
    state_0 = state_0
    fs = pm.Container([f1, f2, f3])

    @pm.deterministic
    def maxf(fs=fs):
        return np.max(fs)

    # rho = pm.Uniform('rho', np.max(fs), 2 * f_top, value=0.1)
    rho = pm.Uniform('rho', maxf + 0.01, f_top, value=0.09)

    @pm.deterministic
    # def sim(rho=rho, om=om, phi=phi, f1=f1, f2=f2, f3=f3, state_0=state_0):
    def sim(rho=rho, om=om, phi=phi, fs=fs, state_0=state_0):
        f = np.concatenate((nums(fs[0], sc[0]), nums(fs[1], sc[1]), nums(fs[2], sc[2])))
        years_prior = 10
        y_0 = difference_model(state_0, r_start - years_prior, r_start,
                               rho, om, phi, f,
                               r=20, full_output=True)
        if type(y_0) != list:
            return -np.inf * np.ones((3, 192))
        state_0 = [yi[:, -1] for yi in y_0]
        r = 3
        y = difference_model(state_0, r_start, r_end,
                             rho, om, phi, f,
                             r=r)

        # Take result and sum values according to susceptibility
        # print(y.shape)
        y = np.split(y, np.cumsum(sc)[:-1])
        y = [yi.sum(axis=0) for yi in y]
        y = [np.sum(yi.reshape(-1, r), axis=1) for yi in y]
        y = np.array(y)
        return y  # This is 3 lists of 192 Monthly points

    @pm.deterministic
    def mu1(sim=sim):
        '''gets monthly data on 3 age groups. Slices to relevant
        data months'''
        # Slices relevant for months data
        start_ix = (1998 - r_start) * 12
        end_ix = (2014 - r_start) * 12
        ret = sim[:, start_ix:end_ix]
        return ret
        # Reduce y to monthly data and take only 1998-2014
        # res = reduce_month(y)[:, start_ix:end_ix]
        # logger.info(res.shape)
        # logger.info(data.shape)

    # @pm.deterministic
    # def mu2(sim=sim):
    #     # start_ix = 1957 - r_start
    #     # end_ix = 2014 - r_start
    #     y = sim.sum(axis=0)
    #     y = np.sum(y.reshape(-1, 12), axis=1)
    #     return y

    Y1 = pm.Normal('Y1', mu=mu1 * p, tau=1 / 50 ** 2, observed=True, value=data1)
    # Y2 = pm.Binomial('Y2', n=mu2, p=p, observed=True, value=data2)

    return locals()
    # return ([rho, om, phi, f, mu1, mu2, Y1, Y2])
