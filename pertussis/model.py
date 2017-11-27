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
    S, Vap, Vwp, Is, Ia, R, Healthy, All, Shots, New = [np.zeros((J, num_months)) for _ in range(10)]  # 9 Matrices
    for i, c in enumerate([S, Vap, Vwp, Is, Ia, R]):
        c[:, 0] = state_0[i]
        All[:, 0] += c[:, 0]
    # New[:, 0] = state_0[9]
    # All[:, 0] = state_0[7]
    # ************************************************* Start Loop ***************************************
    for t, T in enumerate(timeline[1:], start=1):
        demographic_change = 2300 >= T >= data_start
        vax_ap[21] = 0
        t_max = min(2014 - data_start,
                    max(0, int(T) - data_start))  # 2014 where it ends
        # Compartments and Derivatives
        A = All[:, t - 1].sum()
        if A > 30e6:
            print("TOO MUCH", T)
            logger.setLevel(logging.INFO)
            logger.error("Too much people at time: {:.2f}".format(T))
            # return [S, Vap, Vwp, Is, Ia, R, Healthy, All, New]
            A
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
        beta_ = f * beta(T, om, phi, rho)
        # 0 + cos((2 * pi * (t - zero_year) / omega) + phi)
        # rho + f*cos(...)
        # f * (rho + cos(...)) = (f1/f2/f3) * rho + f*cos(...)

        beta_ = np.where(beta_ >= 1e-9, beta_, 1e-9)  # Take only non-negative valuese
        IC = nI.dot(C)
        lambda_ = beta_ * IC
        if any(lambda_ < 0):
            logger.warning('0 likelihood')
            logger.error('0 likelihood lambda<0')
            logger.error("rho {}\nomega {}\nphi {}\nf {} \n\n".format(rho, om, phi, f))
            logger.setLevel(logging.INFO)
            return no_likelihood

        ##### Equations ####
        ## Susceptible
        # Birth
        if demographic_change:
            preg_cover = 0
            if T >= 2016:
                preg_cover = 0.5
            babies = delta[t_max] * r
            Vap[0, t] += babies * preg_cover
            S[0, t] += babies * (1 - preg_cover)  # IN: Birth rate
        # Transmission
        S[:, t] += r * (omega * nR + omega_ap * nVap + omega_wp * nVwp)  # IN: Waning from Natural and vaccine
        S[:, t] -= r * lambda_ * nS  # OUT: Becoming infected
        if any(S[:, t] < 0):
            logger.warning("S < 0 at {} \n {}".format(T, S[:, t]))
            logger.setLevel(logging.ERROR)
            # return no_likelihood
            # print ("S < 0")
        ##### Vaccination
        if data_start <= T < 1957:
            S[1:, t] += r * a * nS[:-1]  # IN
        if 1957 <= T < 2002:  # Begin wp
            Vwp[1:, t] += r * vax_wp * a * nS[:-1]  # IN: Age and vaccinate wP from S classes
            S[1:, t] += r * (1 - vax_wp) * a * nS[:-1]  # IN: Age from previous age no vaccine
        if T >= 2002:  # Begin aP
            if T >= 2015:
                vax_ap[21] = preg_cover * 0.5
            Vap[1:, t] += r * vax_ap * a * nS[:-1]  # IN: Age and vaccinate aP from S classes >>>>>>>>>>
            S[1:, t] += r * (1 - vax_ap) * a * nS[:-1]  # IN: Age from previous age no vaccine
            Shots[1:, t] += r * vax_ap * a * nA[:-1]
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
        Shots[:, t] *= A
        New[:, t] *= A
    Healthy = S + Vap + Vwp
    logger.setLevel(logging.INFO)
    if full_output:
        return [S, Vap, Vwp, Is, Ia, R, Healthy, All, Shots, New]
    return New


def run_model(state_0, start, end,
              om, phi, rho, f1, f2, f3, e=1,
              r=3, r_0=20, years_prior=10):
    no_likelihood = -np.inf * np.ones((3, 192)), -np.inf * np.ones(36), -np.inf * np.ones(270)
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
    tmp_cut = np.arange(int(r / N * (1953 - start)),
                        int(r / N * (1957 - start)))
    # Compute Total in the point, and R in the point - needed age is 5 (point 9)
    R, A = y[5], y[7]
    tmp_R = R[9, tmp_cut]  # 9 is the needed age group
    tmp_A = A[9, tmp_cut]
    mean_sick = (tmp_R / tmp_A).mean()
    # exit()
    # Test condition
    if mean_sick <= 0.0:
        logger.error('Not enough sick kids, {:.2f}'.format(mean_sick))
        return no_likelihood
    # Save last state
    state_z = [yi[:, -1] for yi in y]
    state_z = np.concatenate(state_z)
    y = y[-1]  # NEW SICK is the last one
    # Take result and sum values according to susceptibility
    y = np.split(y, np.cumsum(sc)[:-1])
    y = [yi.sum(axis=0) / A.sum(axis=0) for yi in y]
    # print((y[1]/A.sum(axis=0)).shape)
    y = [np.sum(yi.reshape(-1, r), axis=1) for yi in y]
    y = np.array(y) * 10 ** 5
    # y is only new sick people on 3 age (susceptibiliy change) groups: 3,192

    # Monthly
    # Slices relevant for months data
    start_ix = (1998 - start) * 12
    end_ix = (2014 - start) * 12
    extra_ix = (2017 - start) * 12
    monthly = y[:, start_ix:end_ix]
    extra_monthly = y[:, end_ix:extra_ix].sum(axis=0)

    # Yearly
    # start_ix = 1951 - start
    # end_ix = 1999 - start
    # yearly = y.sum(axis=0)
    # yearly = np.sum(yearly.reshape(-1, 12), axis=1)
    # yearly = yearly[start_ix:end_ix]
    # return (monthly, yearly)  # This is 3 lists of 192 Monthly points and 48 yearly
    # return monthly, state_z
    return monthly, extra_monthly, state_z


def init_mcmc(name, state_0, r_start, r_end, *params, **kwargs):
    import copy
    extra = copy.deepcopy(kwargs)
    mcmc = {'name': name}  # Name- also name of file

    # Constatns
    mcmc['accept_hat'] = 0.23
    mcmc['sigma'] = 100
    mcmc['state_0'] = state_0
    mcmc['start'] = r_start
    mcmc['end'] = r_end

    mcmc['active_params'] = np.array(extra.get('active_params', np.arange(6)))
    apsl = mcmc['active_params']
    mcmc['names'] = extra.get('names', ['omega', 'phi', 'rho', 'f1', 'f2', 'f3', 'e'])
    # mcmc['names'] = np.array(mcmc['names'])[apsl]
    # Initial Values
    mcmc['values'] = np.array(params)[apsl]  # CURRENT values ########################################
    # vals = np.array(params)
    # vals[apsl] = mcmc['values'].copy()
    print(mcmc['values'], params)
    M_now, M2_now, state_z = run_model(state_0, r_start, r_end, *params)

    mcmc['y_now_M'] = M_now.copy()  # [None, :, :]  # CURRENT values ########################################
    mcmc['y2_now_M'] = M2_now.copy()  # [None, :, :]  # CURRENT values ########################################
    mcmc['active'] = True

    # Model Soecific
    mcmc['d'] = len(apsl)  # len(mcmc['names'])
    # mcmc['cov'] = np.diag((1/10, np.pi / 10, 0.5, 0.000005, 0.000005, 0.000005, 0.25 / 10))
    # mcmc['cov'] /= 100
    mcmc['scaling_factor'] = np.array([2.4 / np.sqrt(mcmc['d'])])

    # Chains and Metrics
    mcmc['y_hat_M'] = mcmc['y_now_M'].copy()[None, :, :]  # CHAIN: y values for proposed set (shape like data points)
    mcmc['y2_hat_M'] = mcmc['y2_now_M'].copy()  # CHAIN: y values for proposed set (shape like data points)
    # mcmc['y_hat_Y'] = mcmc['y_now_Y'].copy()  # CHAIN: y values for proposed set (shape like data points)
    mcmc['chain'] = mcmc['values'].copy()  # CHAIN: proposed set
    mcmc['guesses'] = mcmc['values'].copy()  # CHAIN: proposed set

    mcmc['accepted'] = np.array([1])
    mcmc['rates'] = np.array([-1])
    mcmc['max_likelihood'] = -np.inf
    mcmc['gelman_rubin'] = np.zeros(mcmc['d'])
    mcmc['change'] = np.array([-1])
    mcmc['initial_guess'] = np.array(params)
    mcmc['tally'] = 0
    mcmc = {**mcmc, **extra}
    # After Uniting
    # State Z
    mcmc['state_z'] = state_z
    # ll_now = log_liklihood(mcmc['y_now_M'], mcmc['datax'], mcmc['sigma'])
    ll_now = log_liklihood(M_now, mcmc['datay1'], mcmc['sigma'], noise=150)
    ll_now += log_liklihood(M2_now, mcmc['datay2'], mcmc['sigma2'], noise=300)
    mcmc['ll'] = np.array([-np.inf, ll_now])
    # SD
    mcmc['cov'] = mcmc['cov'][apsl, :][:, apsl]
    mcmc['sd'] = mcmc['scaling_factor'] ** 2 * mcmc['cov']
    return mcmc
