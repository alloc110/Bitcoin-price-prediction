import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import torch
import matplotlib
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, median_absolute_error, r2_score, explained_variance_score

# Path data file -- Read data
path_data = "data/init/CBBTCUSD.csv"
df = pd.read_csv(path_data)

# Preprocessing
df["CBBTCUSD"] = df["CBBTCUSD"].interpolate(method='linear')

# Data Visualization

# fig, ax = plt.subplots()
# ax.plot(df["CBBTCUSD"])
# ax.set(ylabel='USD', title='Bitcoin Price')
# ax.grid()
# plt.show()

def LinearModel(df, feature_num):
    # Preprocessing
    for i in range(1, feature_num + 1):
        df['Day i + {}'.format(i)] = df["CBBTCUSD"].shift(-1 * i)
    df = df.rename(columns={'CBBTCUSD': "Day i"})

    df = df[:-feature_num]

    #Train and test
    x = df.drop({'Day i + {}'.format(feature_num), 'observation_date'}, axis=1)
    y = df['Day i + {}'.format(feature_num)]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    test_predict = reg.predict(x_test)
    train_predict = reg.predict(x_train)
    print(y_train, train_predict)
    return reg , y_train,  train_predict, y_test, test_predict

reg3, train_actual, train_predict, test_actual, test_predict = LinearModel(df, 3)

# reg4, x4_test, y4_test , x4, y4 = LinearModel(df, 4)
# y4_predict = reg4.predict(x4)
#
# reg5 ,x5_test, y5_test ,x5 , y5= LinearModel(df, 5)
# y5_predict = reg5.predict(x5)

# REPORT
def regression_report(y_true, y_pred):
    error = y_true - y_pred
    percentil = [5, 25, 50, 75, 95]
    percentil_value = np.percentile(error, percentil)

    metrics = [
        ('mean absolute error', mean_absolute_error(y_true, y_pred)),
        ('median absolute error', median_absolute_error(y_true, y_pred)),
        ('mean squared error', mean_squared_error(y_true, y_pred)),
        ('max error', max_error(y_true, y_pred)),
        ('r2 score', r2_score(y_true, y_pred)),
        ('explained variance score', explained_variance_score(y_true, y_pred))
    ]

    print('Metrics for regression:')
    for metric_name, metric_value in metrics:
        print(f'{metric_name:>25s}: {metric_value: >20.3f}')

    print('\nPercentiles:')
    for p, pv in zip(percentil, percentil_value):
        print(f'{p: 25d}: {pv:>20.3f}')

# print("--5 Features--")
# regression_report(y5, y5_predict)
# print("--3 Features--")
# regression_report(y3, y3_predict)
# print("--4 Features--")
# regression_report(y4, y4_predict)


def plot_results(y_train, train_predict, y_test, test_predict):
    """Plots the actual vs. predicted values."""
    plt.figure(figsize=(15, 6))
    plt.plot(y_train, label='Train Actual',linewidth =2)
    plt.plot(train_predict, label='Train Predicted',linewidth =2)
    plt.plot( y_test, label='Test Actual',linewidth =2)
    plt.plot(np.arange(len(y_train), len(y_train) + len(test_predict)), test_predict, label='Test Predicted',linewidth =2)
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Data Visualization For Report
train_predict  = train_predict.reshape(-1,1)
test_predict = test_predict.reshape(-1,1)

plot_results(train_actual, train_predict, test_actual, test_predict)
# plt.plot(y3)
# plt.plot(y3_predict, color = 'blue', linestyle='--')
# # plt.plot()
# plt.show()
