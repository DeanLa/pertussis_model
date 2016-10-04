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


def cases_monthly(path='./data/_imoh/cases.csv'):
    '''Loads Monthly Data per 1e5'''
    df = pd.read_csv(path)
    try:
        pop = pd.read_csv('./data/demographics/birth_rate.csv')
    except:
        logger.error("Can't find path to population by years")
        return
    pop['Y'] = pop['year']
    pop = pop[['Y', 'population']]
    pop.set_index('Y', inplace=True)
    g = df.groupby(['Y', 'M'])
    df = g.agg('count')[['Age', 'Dt']]
    df = df.join(pop)
    # Create per 100 k
    data = (1e5 / 1e3) * df['Age'] / df['population']
    months = df.index.get_level_values(level=0).values + df.index.get_level_values(level=1).values / 12
    logger.info("{} Monthly data values".format(len(months)))
    return data.values, months


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