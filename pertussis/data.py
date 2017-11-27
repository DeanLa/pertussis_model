import numpy as np
import pandas as pd
from pertussis import *


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
    data_next = pd.read_csv('./data/cases_next.csv')['cases'].values[:36] / p.mean()
    months_next = np.linspace(2014 + 1/12, 2017, 36)
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
        pop_next = np.repeat(pop_next, 12, axis=0)
        print(pop.shape, data.shape)
        print(pop_next.shape, data_next.shape)
        data /= pop
        data_next /= pop_next
        data *= 10 ** 5
        data_next *= 10 ** 5
    # return data, months
    return data, data_next, months, months_next


if __name__ == '__main__':
    pass
