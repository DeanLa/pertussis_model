from pertussis import *


#
# def test_model():
#     del (J)
#     from pertussis import J
#     print (J)

def hetro_model(INP, t,
                o, p, f, zeta, d):
    delta = d
    mu = d

    T = reduce_time(t, step=1 / N)

    ## Compartments and Derivatives
    S, Vap, Vwp, Is, Ia, R = unpack(INP, *unpack_values)
    A = S + Vap + Vwp + Is + Ia + R
    # print (A)
    dS, dVap, dVwp, dIs, dIa, dR = (np.zeros(uv) for uv in unpack_values)  # Zeros by size

    ## Helpers and Pre-Calculations
    # TODO: lambda_a needs to be addressed from here
    I = Ia + Is  # Helper
    beta_ = beta(T, o, p) * f
    IC = I.dot(C)
    IsC = Is.dot(C)
    IaC = Ia.dot(C)
    # print (beta_.shape, IC.shape)
    lambda_ = beta_ * IC  # Needs to be normalized
    # lambda_s = beta_ * IaC
    # lambda_a = beta_ * IsC * zeta
    # lambda_ = lambda_s + lambda_a
    e_ap = 1 - epsilon_ap  # Helper
    e_wp = 1 - epsilon_wp  # Helper

    ## Equations

    # Susceptible
    # print (int(T) - 1948)
    dS[0] = delta[int(T) - 1948]  # IN: Birth rate
    dS += omega * R + omega_ap * Vap + omega_wp * Vwp  # IN: Waning from Natural and vaccine
    dS -= lambda_ * S  # OUT: Becoming infected

    # Vaccination
    if T < 1957:
        dS[1:] += S[:-1] * a  # IN

    if 2002 > T >= 1957:  # Begin wp
        dVwp[1:1 + n_wp] += S[:n_wp] * a[:n_wp]  # IN: Age and vaccinate wP from S classes
        dS[1 + n_wp:] += S[n_wp:-1] * a[n_wp:]  # IN: Age from previous age no vaccine
        pass
    if T >= 2002:  # Begin aP
        dVap[1:1 + n_ap] += S[:n_ap] * a[:n_ap]  # IN: Age and vaccinate aP from S classes
        dS[1 + n_ap:] += S[n_ap:-1] * a[n_ap:]  # #IN: Age from previous age no vaccine

    # dS[1:] += S[:-1] * a  # IN
    dVap -= lambda_ * e_ap * Vap  # OUT: Getting sick (reduced by efficacy)
    dVwp -= lambda_ * e_wp * Vwp  # OUT: Getting sick (reduced by efficacy)
    dVap -= Vap * omega_ap  # OUT: Waning
    dVwp -= Vwp * omega_wp  # OUT: Waning

    # Infected
    infected_ap = lambda_ * e_ap * Vap  # HELPER: Infected with ap
    infected_wp = lambda_ * e_wp * Vwp + lambda_ * S  # HELPER: Infected with wp or no vaccine

    dIs = alpha_ap * infected_ap + alpha_wp * infected_wp  # IN: Infected with symptoms chance
    dIs -= gamma_s * Is  # OUT: Recovered
    # dIs += M

    dIa = (1 - alpha_ap) * infected_ap + (1 - alpha_wp) * infected_wp  # IN: Infected with NO symptoms chance
    dIa -= gamma_a * Ia  # OUT: Recovered
    # dIa += M

    # Recovered
    dR = gamma_s * Is + gamma_a * Ia  # IN: Recovered from I
    dR -= omega * R  # OUT: Natrual waning
    # dR -= 2 * M

    ## Regular Aging
    # S
    dS[:-1] -= S[:-1] * a  # OUT: Age out to Next V or Next S, OUT is the same
    # V
    dVap[1:] += Vap[:-1] * a  # IN
    dVap[:-1] -= Vap[:-1] * a  # OUT
    dVwp[1:] += Vwp[:-1] * a  # IN
    dVwp[:-1] -= Vwp[:-1] * a  # OUT
    # I and R
    dR[1:] += R[:-1] * a  # IN
    dR[:-1] -= R[:-1] * a  # OUT
    dIs[1:] += Is[:-1] * a  # IN
    dIs[:-1] -= Is[:-1] * a  # OUT
    dIa[1:] += Ia[:-1] * a  # IN
    dIa[:-1] -= Ia[:-1] * a  # OUT
    ## Housekeeping
    Y = pack_flat((dS, dVap, dVwp, dIs, dIa, dR))
    # Older people die more
    '''I fix the death rate so that old people die more ofern
    If this isn't adjusted to oldest age will explode. We manually fit this to data'''
    mu = mu[int(T) - 1948]
    r = 0.7
    mu_v = np.ones(J) * mu * r
    mu_v[-1] += (1 - r) * mu * J
    s = (mu_v * A).sum()
    # print ("S", s, mu, mu/s)
    mu_v = np.tile(mu_v, 6) * mu / s
    Y -= mu_v * INP  # OUT: Death
    # print (Y)
    # Y[-1] -= delta[int(T) - 1948]
    return Y


def hetro_model_relative(INP, t,
                         o, p, f):
    T = reduce_time(t, step=1 / N)
    ## Compartments and Derivatives
    S, Vap, Vwp, Is, Ia, R = unpack(INP, *unpack_values)
    A = S + Vap + Vwp + Is + Ia + R
    dS, dVap, dVwp, dIs, dIa, dR = (np.zeros(uv) for uv in unpack_values)  # Zeros by size

    ## Helpers and Pre-Calculations
    # TODO: lambda_a needs to be addressed from here
    I = Ia + Is  # Helper
    beta_ = beta(T - 1948, o, p) * f
    IC = I.dot(C)
    lambda_ = beta_ * IC  # Needs to be normalized
    # TODO: lambda_a - be more infectious on asymptomatic
    lambda_a = zeta * lambda_
    e_ap = 1  # - epsilon_ap  # Helper
    e_wp = 1  # - epsilon_wp  # Helper

    ## Equations

    # Susceptible
    dS[0] = delta[int(T) - 1948]  # IN: Birth rate
    dS += omega * R + omega_ap * Vap + omega_wp * Vwp  # IN: Waning from Natural and vaccine
    dS -= lambda_ * S  # OUT: Becoming infected

    # Vaccination
    if T < 1957:
        dS[1:] += S[:-1] * a  # IN

    if 2002 > T >= 1957:  # Begin wp
        dVwp[1:1 + n_wp] += S[:n_wp] * a[:n_wp]  # IN: Age and vaccinate wP from S classes
        dS[1 + n_wp:] += S[n_wp:-1] * a[n_wp:]  # IN: Age from previous age no vaccine
        pass
    if T >= 2002:  # Begin aP
        dVap[1:1 + n_ap] += S[:n_ap] * a[:n_ap]  # IN: Age and vaccinate aP from S classes
        dS[1 + n_ap:] += S[n_ap:-1] * a[n_ap:]  # #IN: Age from previous age no vaccine

    # dS[1:] += S[:-1] * a  # IN
    dVap -= lambda_ * e_ap * Vap  # OUT: Getting sick (reduced by efficacy)
    dVwp -= lambda_ * e_wp * Vwp  # OUT: Getting sick (reduced by efficacy)
    dVap -= Vap * omega_ap  # OUT: Waning
    dVwp -= Vwp * omega_wp  # OUT: Waning

    # Infected
    infected_ap = lambda_ * e_ap * Vap  # HELPER: Infected with ap
    infected_wp = lambda_ * e_wp * Vwp + lambda_ * S  # HELPER: Infected with wp or no vaccine

    dIs = alpha_ap * infected_ap + alpha_wp * infected_wp  # IN: Infected with symptoms chance
    dIs -= gamma_s * Is  # OUT: Recovered
    dIs += M

    dIa = (1 - alpha_ap) * infected_ap + (1 - alpha_wp) * infected_wp  # IN: Infected with NO symptoms chance
    dIa -= gamma_a * Ia  # OUT: Recovered
    dIa += M

    # Recovered
    dR = gamma_s * Is + gamma_a * Ia  # IN: Recovered from I
    dR -= omega * R  # OUT: Natrual waning
    dR -= 2 * M

    ## Regular Aging
    # S
    dS[:-1] -= S[:-1] * a  # OUT: Age out to Next V or Next S, OUT is the same
    # V
    dVap[1:] += Vap[:-1] * a  # IN
    dVap[:-1] -= Vap[:-1] * a  # OUT
    dVwp[1:] += Vwp[:-1] * a  # IN
    dVwp[:-1] -= Vwp[:-1] * a  # OUT
    # I and R
    dR[1:] += R[:-1] * a  # IN
    dR[:-1] -= R[:-1] * a  # OUT
    dIs[1:] += Is[:-1] * a  # IN
    dIs[:-1] -= Is[:-1] * a  # OUT
    dIa[1:] += Ia[:-1] * a  # IN
    dIa[:-1] -= Ia[:-1] * a  # OUT
    ## Housekeeping
    Y = pack_flat((dS, dVap, dVwp, dIs, dIa, dR))
    Y -= mu[int(T) - 1948] * INP  # OUT: Death
    return Y


def vaccine_model(INP, t, step, m1, omega_, phi):
    from z_old.params import collect_params
    T = reduce_time(t, step=step)
    M = 1e-6

    ## Compartments
    S, Vap, Vwp, Is, Ia, R = unpack(INP, 1, 6, 4, 1, 1, 1)
    ## Params
    delta, lambda_s, lambda_a, gamma_a, \
    gamma_s, omega, mu, alpha_ap, \
    alpha_wp, c, phi_ap, phi_wp, \
    epsilon_ap, epsilon_wp, omega_ap, omega_wp = \
        collect_params(T, step)

    lambda_ = beta(T - 1948, m1, omega_, phi)
    # lambda_ = 0.5

    e_ap = 1 - epsilon_ap
    e_wp = 1 - epsilon_wp
    # Is += 0.1
    I = Ia + Is  # Helper

    ## Equations

    # Susceptible
    dS = delta  # IN: Birth rate
    dS += omega * R + Vap.sum() * omega_ap + Vwp.sum() * omega_wp  # IN: Waning from Natural and vaccine
    dS -= lambda_ * I * S  # OUT: Becoming infected

    # Vaccinated
    dVap = np.zeros(6)
    dVwp = np.zeros(4)
    if T >= 1957:  # Begin wp
        if T < 2002:
            dS -= phi_wp[0] * S  # OUT: Age to wp0 after 1957 before 2002
            dVwp[0] = phi_wp[0] * S  # S to wp0 if before 2002
        dVwp[0] -= (phi_wp[1] + lambda_ * I * e_wp[0]) * Vwp[0]  # Get next vaccine
        for i in range(1, len(dVwp) - 1):
            dVwp[i] = phi_wp[i] * Vwp[i - 1]  # IN: previous Vwp
            dVwp[i] -= (phi_wp[i + 1] + lambda_ * I * e_wp[i]) * Vwp[i]  # OUT: to infected and next vaccine
        dVwp[-1] = phi_wp[-1] * Vwp[-2]  # IN: previous Vwp
        dVwp[-1] -= lambda_ * I * e_wp[-1] * Vwp[-1]  # OUT: to Infected

    if T >= 2002:  # Begin aP
        dS -= phi_ap[0] * S  # OUT: Age to ap after 2002
        dVap[0] = phi_ap[0] * S  # IN: age from S
        dVap[0] -= (phi_ap[1] + lambda_ * I * e_ap[0]) * Vap[0]  # OUT: age to next, and get infected
        for i in range(1, len(dVap) - 1):
            dVap[i] = phi_ap[i] * Vap[i - 1]  # IN: age from previous Vap
            dVap[i] -= (phi_ap[i + 1] + lambda_ * I * e_ap[i]) * Vap[i]  # OUT: to infected and next vaccine
        dVap[-1] = phi_ap[-1] * Vap[-2]  # IN: previous Vwp
        dVap[-1] -= lambda_ * I * e_ap[-1] * Vap[-1]  # OUT: to Infected

    dVap -= Vap * omega_ap  # Waning
    dVwp -= Vwp * omega_wp  # Waning

    # Infected
    infected_ap = lambda_ * I * (e_ap * Vap).sum()  # HELPER: Infected with ap
    infected_wp = lambda_ * I * (e_wp * Vwp).sum() + lambda_ * I * S  # HELPER: Infected with wp or no vaccine

    dIs = alpha_ap * infected_ap + alpha_wp * infected_wp  # IN: Infected with symptoms chance
    dIs -= gamma_s * Is  # OUT: Recovered
    dIs += M

    dIa = (1 - alpha_ap) * infected_ap + (1 - alpha_wp) * infected_wp
    dIa -= gamma_a * Ia  # OUT: Recovered
    dIa += M

    # Recovered
    dR = gamma_s * Is + gamma_a * Ia - omega * R - 2 * M

    ## Housekeeping
    Y = pack((dS, dVap, dVwp, dIs, dIa, dR))
    Y -= mu * INP  # OUT: Death
    return Y


def base_model(INP, t, step):
    T = reduce_time(t, step=step)
    print(T)
    S1, Va, Is, Ia, R = INP
    delta, lambda_s, lambda_a, gamma_a, \
    gamma_s, omega, mu, alpha, \
    alpha_tag, c, aP, wP = \
        collect_params(T, step)

    # Susceptible 1
    dS1 = delta - (lambda_s * Is + lambda_a * Ia) * S1 + omega * R

    # Vaccinated
    dVa = - (lambda_s * Is + lambda_a * Ia) * Va

    dIs = (lambda_s * Is + lambda_a * Ia) * (alpha * S1 + alpha_tag * Va) \
          - gamma_s * Is

    dIa = (lambda_s * Is + lambda_a * Ia) * ((1 - alpha) * S1 + (1 - alpha_tag) * Va) \
          - gamma_a * Ia

    dR = gamma_s * Is + gamma_a * Ia - omega * R

    # if T <= 1957:
    #     # No Vaccine
    #     pass
    # if T <= 2002 and T > 1957:
    #     # WCV
    #     dS1 -= c * mu
    #     dR += c * mu
    # if T > 2002:
    #     dS1 -= c * mu
    #     dVa += c * mu

    Y = np.array([dS1, dVa, dIs, dIa, dR])
    Y -= mu * INP
    return Y
