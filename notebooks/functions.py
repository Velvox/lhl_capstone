import os
import requests as re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

target_column = 'nflx_Adj Close'

## APIs

def gNewsSearch(date_start, date_end, query='netflix company', lang='en'):
    """
    Takes a query and timeframe to return a set of articles,
    requires api key stores as GNEWS_API_KEY in OS var.
    Returns a response object.
    
    query (str)
    date_start (string) YYYY-MM-DDTHH:MM:SSZ
    date_end (string) YYYY-MM-DDTHH:MM:SSZ
    lang (str)
    """
    
    api_key = os.environ["GNEWS_API_KEY"]

    url = 'https://gnews.io/api/v4/search'

    params = {
        'token': api_key,
        'q': query,
        'from': date_start,
        'to': date_end,
        'lang': lang
    }

    response = (re.get(url, params=params))
    return response

## Rolling mean prediction

def rolling_mean_prediction(series, window):
    """
    Takes a time series and a window (int) to predict the value that follows the window.
    returns an array of predictions and prints plots

    series (pandas Series)
    window (int)
    """
    X = series.values
    window=window
    history = [X[i] for i in range(window)]
    test = [X[i] for i in range(window, len(X))]
    predictions = list()
    # walk forward over time steps in test
    for t in range(len(test)):
        length = len(history)
        yhat = np.mean([history[i] for i in range(length-window,length)])
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.show()
    # zoom plot
    plt.plot(test[0:100])
    plt.plot(predictions[0:100], color='red')
    plt.show()
    return predictions

## Feature engineering

def concatRename(df1, name1, df2, name2):
    """
    Concatenates and appends a specific prefix to each column name.
    df1 (dataframe)
    name1 (string)
    df2 (dataframe)
    name2 (string)
    """
    df1.columns = [name1 + name for name in df1.columns]
    df2.columns = [name2 + name for name in df2.columns]
    df3 = pd.concat([df1,df2], axis=1)
    return df3

def testSplit(df, test_size=0.20):
    """
    Takes a dataframe and removes the last 20% of rows for testing, specific for time series.
    Returns (train, test).

    df (dataframe)
    test_size (float)
    """
    data = df.copy()
    rows = data.shape[0]
    n_drop = int(rows * test_size)
    test = data.tail(n_drop)
    data.drop(data.tail(n_drop).index, inplace = True)
    return data, test

def dropNa(df):
    """
    Drops rows with NA values, returns df
    """
    data = df.copy()
    data = data.dropna(axis=1)
    return data

def dropTarget(df, column=target_column):
    data = df.copy()
    data = data.drop(column, axis=1)
    return data

# shift functions for entire df

def shiftTime(df, rolling=1):
    """
    Takes a dataframe and column target name to return a df with a new lagged value column
    """
    data = df.copy()
    data = data.shift(rolling)
    return data

def rollingMeanShift(df, column=target_column, rolling=3, period=1):
    """
    Takes a df, column name string, rolling window int and period int to create a new time series feature column.
    """
    data = df.copy()
    data = data.rolling(rolling).mean().shift(period)
    return data

def trendDiff(df, column=target_column, delta=1):
    """
    Calculates a trend feature for a given timeframe.
    """
    data = df.copy()
    data = data - data.shift(delta)
    return data

# shift functions for additionnal features

def shiftTime_col(df, column=target_column, rolling=7):
    """
    Takes a dataframe and column target name to return a df with a new lagged value column
    """
    data = df.copy()
    data[column+'_lag'+str(rolling)] = data[column].shift(rolling)
    return data

def rollingMeanShift_col(df, column=target_column, rolling=7, period=1):
    """
    Takes a df, column name string, rolling window int and period int to create a new time series feature column.
    """
    data = df.copy()
    data[column+'_MA'+str(rolling)+'_lag'+str(period)] = data[column].rolling(rolling).mean().shift(period)
    return data

def trendDiff_col(df, column=target_column, delta=1):
    """
    Calculates a trend feature for a given timeframe.
    """
    data = df.copy()
    data[column+'_delta'+str(delta)] = data[column] - data[column].shift(delta)
    return data