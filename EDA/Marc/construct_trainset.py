import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt


def get_average_order_amount_user_item(_orders):
    res = _orders[['userID', 'itemID', 'order']].groupby(['userID','itemID']).mean().reset_index()
    res.rename(columns={'order':'avg_order_user_item'}, inplace=True)
    return res


def get_average_order_amount_item(_orders):
    res = _orders[['itemID', 'order']].groupby(['itemID']).mean().reset_index()
    res.rename(columns={'order':'avg_order_item'}, inplace=True)
    return res

def get_average_order_amount_user(_orders):
    res = _orders[['userID', 'order']].groupby(['userID']).mean().reset_index()
    res.rename(columns={'order':'avg_order_user'}, inplace=True)
    return res

def number_orders_user_item(_orders):
    return _orders[['userID', 'itemID']].groupby(['userID', 'itemID']).size().reset_index().rename(columns={0:'num_orders_user_item'})


def feature_avg_lifespan(_orders):
    workcopy = _orders[['date','userID', 'itemID']].copy()
    workcopy.sort_values(['userID', 'itemID', 'date'], inplace=True)
    workcopy['diffs'] = workcopy.groupby(['userID', 'itemID'])['date'].diff()
    workcopy = workcopy[pd.notnull(workcopy['diffs'])]
    workcopy['sum'] = 1
    workcopy['diffs'] = workcopy.diffs.dt.days
    workcopy = workcopy[['userID', 'itemID', 'diffs', 'sum']].groupby(['itemID', 'userID']).sum().reset_index()
    workcopy = workcopy.drop(columns=['userID']).groupby('itemID').sum().reset_index()
    workcopy['avg_lifespan'] = workcopy['diffs'] / workcopy['sum']
    return workcopy.drop(columns=['sum', 'diffs'])


def lastpurchase(_orders, date):
    date = pd.Timestamp(date)
    res = _orders[['userID', 'itemID', 'date']].groupby(['userID', 'itemID']).max().reset_index()
    res['last_purchased'] = res['date'].apply(lambda d: (date - d))
    res['last_purchased'] = res.last_purchased.dt.days
    res.drop(columns=['date'], inplace=True)
    return res


def create_train_labels(X, month):
    #set month in each example
    X['month'] = X['date'].dt.month
    #set week in month. We want at most 4
    X['week'] = X.date.apply(lambda d: int(min((d.day-1)//7 + 1, 4)))
    #every existing combination of userID and itemID
    user_items = X[['userID', 'itemID']].groupby(['userID', 'itemID']).all().reset_index()
    user_items['week'] = 5
    months = pd.DataFrame({
        'month':[month]
    })
    #every combination of existing userIDxitemID pairs with every month found in the data
    user_items = user_items.merge(months, how='cross')
    user_item_month_week = X[['month', 'userID', 'itemID', 'week']]
    user_item_month_week = pd.concat([user_item_month_week, user_items])
    '''find for every userxitemxmonth combination the min week. If the combination doesn't exist in the data, 
    the week will be 5 since we added all those combinations in the step before
    Afterwards we set week to 0 where week=5, since these combinations weren't present in the data
    '''
    res = user_item_month_week.groupby(['month', 'userID', 'itemID']).min().reset_index()
    res['week'] = res.week.apply(lambda w: w if w < 5 else 0)
    return res

def create_trainset(_orders, cutoffdate):
    date = pd.Timestamp(cutoffdate)
    orders_before_cutoff = _orders[_orders['date'] < date]
    relevant_user_item_combs = orders_before_cutoff.groupby(['userID', 'itemID']).counts()
    relevant_user_item_combs = relevant_user_item_combs.filter(lambda x: len(x)>=2)
    relevant_orders = _orders.merge(relevant_user_item_combs, how='inner', on=['userID', 'itemID'])
    labels = create_train_labels(relevant_orders, date.month)
    train_features = relevant_user_item_combs
    for f in [get_average_order_amount_user_item, get_average_order_amount_item, get_average_order_amount_user, number_orders_user_item, feature_avg_lifespan, lastpurchase]:
        train_features = train_features.merge(f(orders_before_cutoff, how='left'))
    train_data = labels.merge(train_features, how='left', on=['userID', 'itemID'])
    train_data = train_data.drop(columns=['userID', 'itemID'])
    return (train_data.drop(columns='week'), train_data['week'])
    
    