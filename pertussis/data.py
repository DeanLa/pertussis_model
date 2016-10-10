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


def cases_yearly(path='./data/yearly.csv'):
    '''Loads the yearly data per 1e5'''
    data = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 1]
    years = np.genfromtxt(path, delimiter=',', skip_header=1)[:, 0]
    logger.info("{} Yearly data values".format(len(years)))
    return data, years


def cases_monthly(path='./data/_imoh/cases.csv', collapse = False):
    '''Loads Monthly Data in total numbers
    '''
    df = pd.read_csv(path)
    sc_len = len(sc)
    for i in range(sc_len):
        col = "sc_{}".format(sc_ages[i])
        df[col] = (sc_ages[i] <= df['Age']) & (df['Age'] < sc_ages[i+1])
        df[col] = df[col].astype(int)
    g = df.groupby(['Y', 'M'])
    df = g.agg('sum')#[['Age', 'Dt']]
    data = df.ix[:,-sc_len:]
    # print (len(data))
    # sys.exit('inside data/cases_monthly')

    months = df.index.get_level_values(level=0).values + df.index.get_level_values(level=1).values / 12
    logger.info("{x[0]} Monthly data values on {x[1]} age groups".format(x=data.shape))
    if collapse:
        data = data.sum(axis=1)
    return data.values, months
    # try:
    #     pop = pd.read_csv('./data/demographics/birth_rate.csv')
    # except:
    #     logger.error("Can't find path to population by years")
    #     return
    # pop['Y'] = pop['year']
    # pop = pop[['Y', 'population']]
    # pop.set_index('Y', inplace=True)
    # df = df.join(pop)
    # Create per 100 k


def cases_month_age(path='./data/_imoh/cases.csv'):
    df = pd.read_csv(path, ).fillna(-1000)
    xp100 = pd.read_csv('./data/demographics/pop_by_year_1998_2014.csv')
    x = df.merge(xp100, on="Y")
    x['YM'] = x.Y + (x.M - 1) / 12
    # print (x.T)
    # x['W'] = 1e-3 * ((x.Age < 20) * young_factor + (x.Age >= 20) * old_factor) / x.population # Weights
    x['W'] = 1e-3 * 1 / x.population # Weights
    bins_ages = np.append(a_l, 120)
    bins_time = np.arange(1998, 2014.01, 1 / 12)
    h = np.histogram2d(x.Age, x.YM, bins=[bins_ages, bins_time], weights=x.W)

    return h[0]

if __name__ == '__main__':
    pass