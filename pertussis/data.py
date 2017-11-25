import numpy as np
import pandas as pd
from pertussis import *


def get_dist_98():
    return np.array([0.003854352, 0.003854352, 0.003854352,
                     0.011563056, 0.118140906, 0.111728033,
                     0.123872792, 0.084315304, 0.26262968,
                     0.175186235, 0.101000938])


def get_death_rate(path='./data/demographics/death_rate.csv'):
    death_rate = pd.read_csv(path)
    death_rate.interpolate(method='polynomial', order=2, inplace=True)
    for col in _group_names:
        poly_extrapolate(death_rate, 'Year', col, deg=1)
    death_rate = death_rate.set_index('Year')
    return death_rate


def cases_yearly(path='./data/year_sick.csv', reporting_rate=p):
    '''Loads the yearly data per 1e5'''
    data = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 3]
    pop = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 2] * 1000
    years = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
    sl = (years >= 1951) & (years <= 1998)
    data = data[sl]
    years = years[sl]
    pop = pop[sl]
    data /= reporting_rate  # Reporting rate
    datan = data / pop
    # Compute ratios
    deciding_year = 1971
    m1 = datan[years <= deciding_year].mean()
    m2 = datan[years > deciding_year].mean()
    mr = 5 * (m2 / m1)
    print(m1, m2, mr)
    # Change <1975 data to be 5 times more then later
    data[years <= deciding_year] *= mr
    logger.info("{} Yearly data values".format(len(years)))
    return data, datan, years


def cases_monthly(path='./data/cases.csv', collapse=False, per100k=True):
    '''Loads Monthly Data in total numbers and normalized for reporting rate `p`
    '''
    # 1998-2014
    df = pd.read_csv(path)
    sc_len = len(sc)
    for i in range(sc_len):
        col = "sc_{}".format(sc_ages[i])
        df[col] = (sc_ages[i] <= df['Age']) & (df['Age'] < sc_ages[i + 1])
        df[col] = df[col].astype(int)
    g = df.groupby(['Y', 'M'])
    df = g.agg('sum')  # [['Age', 'Dt']]
    data = df.ix[:, -sc_len:]
    months = df.index.get_level_values(level=0).values + df.index.get_level_values(level=1).values / 12
    logger.info("{x[0]} Monthly data values on {x[1]} age groups".format(x=data.shape))
    data = data.values.T
    data = data / p  # Real cases after involving reporting rate
    # 2014 - 2017
    data_next = pd.read_csv('./cases_next.csv')['cases'].values / p.mean()
    if collapse:
        data = data.sum(axis=1)
    if per100k:
        pop = np.genfromtxt('./data/demographics/pop_by_year_1998_2014.csv',
                            delimiter=',', skip_header=1, usecols=1)
        pop_next = np.genfromtxt('./data/demographics/pop_by_year_2014_2017.csv',
                                 delimiter=',', skip_header=1, usecols=1)

        pop *= 10 ** 3
        pop_next *= 10 ** 3
        pop = np.repeat(pop, 12, axis=0)
        pop_next = np.repeat(pop, 12, axis=0)
        print(pop.shape, data.shape)
        print(pop_next.shape, data_next.shape)
        data /= pop
        data_next /= pop_next
        data *= 10 ** 5
        data_next *= 10 ** 5
    # exit()
    # return data, months
    return data, data_next, months


# def cases_month_age(path='./data/_imoh/cases.csv'):
#     df = pd.read_csv(path, ).fillna(-1000)
#     xp100 = pd.read_csv('./data/demographics/pop_by_year_1998_2014.csv')
#     x = df.merge(xp100, on="Y")
#     x['YM'] = x.Y + (x.M - 1) / 12
#     # print (x.T)
#     # x['W'] = 1e-3 * ((x.Age < 20) * young_factor + (x.Age >= 20) * old_factor) / x.population # Weights
#     x['W'] = 1e-3 * 1 / x.population # Weights
#     bins_ages = np.append(a_l, 120)
#     bins_time = np.arange(1998, 2014.01, 1 / 12)
#     h = np.histogram2d(x.Age, x.YM, bins=[bins_ages, bins_time], weights=x.W)
#
#     return h[0]

if __name__ == '__main__':
    pass
