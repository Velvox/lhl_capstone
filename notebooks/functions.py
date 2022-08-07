import os
import requests as re
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_squared_error

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

def rolling_mean_prediction(series, window):
    """
    Takes a time series and a window (int) to predict the value that follows the window.
    returns an array of predictions and prints plots
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

def testSplit(df):
    """
    Takes a dataframe and removes the last 20% of rows for testing, specific for time series.
    Returns (train, test).
    """
    data = df.copy()
    rows = data.shape[0]
    n_drop = int(rows * 0.20)
    test = data.tail(n_drop)
    data.drop(data.tail(n_drop).index, inplace = True)
    return data, test