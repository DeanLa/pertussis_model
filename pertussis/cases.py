import numpy as np
import pandas as pd


def get_death_rate(path = './data/demographics/death_rate.csv'):
    death_rate = pd.read_csv(path)
    death_rate.interpolate(method='polynomial', order=2, inplace=True)
    for col in _group_names:
        poly_extrapolate(death_rate, 'Year', col, deg=1)
    death_rate = death_rate.set_index('Year')
    return death_rate

def cases_yearly():
    pass

def cases_monthly(path='./data/cases.csv'):
    df = pd.read_csv(path)
    pop = pd.read_csv('./data/demographics/birth_rate.csv')
    pop['Y'] = pop['year']
    pop = pop[['Y','population']]
    g = df.groupby(['Y','M'])
    df = g.agg('count')[['Age','Dt']]
    # months =
    return df