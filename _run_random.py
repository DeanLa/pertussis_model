# coding: utf-8
import multiprocessing
from pertussis import *
def random_search(iterations):
    # from pertussis import *

    # logger.setLevel(logging.INFO)
    # Initial
    r_start = 1929
    r_end = 2017
    step = 1 / N

    # Data
    data_M, data_M2, months, months2 = cases_monthly()
    state_0 = collect_state0()
    sigma1 = np.array([50, 100, 100]).reshape(3, 1)
    sigma2 = 100

    # ## Distributions
    dist_om = uniform(3.5, 1)
    dist_phi = uniform(0, 2 * np.pi)
    dist_rho = uniform(10, 100)
    dist_f1 = uniform(0.00, 0.001)
    dist_f2 = uniform(0.00, 0.001)
    dist_f3 = uniform(0.00, 0.0001)
    dist_e = uniform(0.5, 0.5)
    dists = [dist_phi, dist_rho, dist_f1, dist_f2, dist_f3]
    path = './random_search/mp_{}_{}'.format(np.random.rand(),file_name.replace('.log','.csv'))
    f = open(path,'a')
    f.write('ll,om,phi,rho,f1,f2,f3\n')
    f.close()
    print('Start\n')
    for i in range (iterations):
        f = open(path, 'a')
        guess = [3.9805] + [dist.rvs() for dist in dists]
        try:
            y_star_M, y2_star_M, state_z = run_model(state_0, r_start, r_end, *guess, e=1,
                                                     r_0=40)  # RUN MODEL =============================================
            #                 logger.info(str(y_star_M))
            ll_star = log_liklihood(y_star_M, data_M, sigma1)
            ll_star += log_liklihood(y2_star_M, data_M2, sigma2)
        except:
            logger.error('exception at model {} PROBABLY S-I-R fail'.format(mcmc['name']))
            ll_star = -np.inf
        result = "{:.2f},".format(ll_star) + ','.join('{:.7f}'.format(g) for g in guess)
        print (result)
        if ll_star > -25:
            print('V' * 50 + '\nGOOD RESULT\n' + '-' * 50 + '\n')
        f.write(result + '\n')
        f.close()

if __name__ == '__main__':
    pass
    from pertussis import *
    n = 2
    pool = multiprocessing.Pool(n)
    ret = pool.map(random_search, n*[5])
    pool.close()
    pool.join()
