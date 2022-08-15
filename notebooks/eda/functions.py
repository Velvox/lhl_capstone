import os
from re import L
import requests as re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

import sklearn.metrics as metrics

# text packages
from nltk.tokenize import word_tokenize

target_column = 'nflx_Adj Close'

## APIs

def gNewsSearch(date_start: str, date_end: str, query: str='netflix company', lang:str ='en'):
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

def rolling_mean_prediction(series: pd.Series, window: int):
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

def concatRename(df1: pd.DataFrame, name1: str, df2: pd.DataFrame, name2: str):
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

def testSplit(df: pd.DataFrame, test_size: float=0.20):
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

def dropTarget(df: pd.DataFrame, column: str=target_column):
    """
    Drops a specific column
    """
    data = df.copy()
    data = data.drop(column, axis=1)
    return data

# shift functions for entire df

# class shiftTime(BaseEstimator,TransformerMixin):

#     def __init__(self, rolling: int=1):
#         self.rolling = rolling

#     def fit(self, X, rolling=1):
#         return self

#     def transform(self, X, rolling=1):
#         """
#         Takes a dataframe and column target name to return a df with a new lagged value column
#         """
#         data = X.copy()
#         data = data.shift(rolling)
#         return pd.DataFrame(data)
    

def shiftTime(df:pd.DataFrame, rolling: int=1):
    """
    Takes a dataframe and column target name to return a df with a new lagged value column.
    
    Parameters:
    df(dataframe)
    rolling(int): amount of timesteps to shift

    Returns:
    df(dataframe)
    """
    data = df.copy()
    data = data.shift(rolling)
    return data

def rollingMeanShift(df, rolling=3, period=1):
    """
    Takes a df, column name string, rolling window int and period int to create a new time series feature column.

    Parameters:
    df(dataframe)
    rolling(int): amount of timesteps to shift

    Returns:
    df(dataframe)
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

# Text processing

def newsapiArticleUnpack(df: pd.DataFrame):
    """
    Takes a raw News API dataframe and returns it restructured for use.

    df (dataframe)

    Returns:
    df (dataframe)
    """

    # instantiate variables
    index = 0
    articles_dict = {}

    # load articles only
    articles_df=pd.DataFrame(df.loc['articles'])
    
    # find and store in dict
    for date in articles_df.index:
        for item in articles_df.loc[date]:
            for article in item:
                articles_dict[index] = {}
                for key in article.keys():
                    articles_dict[index][key] = article[key]
                index += 1
    
    # turn dict into DF
    result = pd.DataFrame(articles_dict).T
    
    return result

def text_preprocessing(df):
    """
    Takes in a df of text features and processes them
    """
    data = df
    columns = data.columns
    preprocessed_data = {}
    for column in columns:
        temp_list = []
        for text in data[column]:

            # split into words
            tokens = word_tokenize(text)
            
            # lowercase
            tokens = [w.lower() for w in tokens]
            
            # remove punctuation
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]

            # remove stopwords
            stop_words = stopwords.words('english')
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]

            # lemmatize data
            lemmatizer = WordNetLemmatizer()
            lemmatized = [lemmatizer.lemmatize(word) for word in words]
            lemmatized = ' '.join(lemmatized)
            temp_list.append(lemmatized)
        preprocessed_data[column + '_processed'] = temp_list

    preprocessed_df = pd.DataFrame(preprocessed_data)
    return preprocessed_df

def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score