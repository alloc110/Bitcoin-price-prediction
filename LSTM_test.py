import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from triton.language import dtype
import LSTM_MODEL
import sys
import csv



device = 'cuda'
data_train_pathfile = 'data/train/data.csv'
label_train_pathfile = 'data/train/label.csv'
data_test_pathfile = 'data/test/data.csv'
label_test_pathfile = 'data/test/label.csv'
best_model_pathfile = 'train_model_hourly/best_model_4.pth'
scaler_path = 'scaler/scaler.pkl'

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

def plot_results(y_train, train_predict, y_test, test_predict):
    """Plots the actual vs. predicted values."""
    plt.figure(figsize=(15, 6))
    plt.plot(y_train, label='Train Actual')
    plt.plot(train_predict, label='Train Predicted')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Actual')
    plt.plot(np.arange(len(y_train), len(y_train) + len(test_predict)), test_predict, label='Test Predicted')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def test(model , scaler, x_train_tensor, x_test_tensor ):
    with torch.no_grad():
        train_predict_tensor = model(x_train_tensor).squeeze().to('cpu')
        test_predict_tensor = model(x_test_tensor).squeeze().to('cpu')
    train_predict = scaler.inverse_transform(train_predict_tensor.numpy().reshape(-1, 1))
   # test_predict = scaler.inverse_transform(test_predict_tensor.numpy().reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict_tensor.numpy().reshape(-1, 1))

    return train_predict, test_predict

data_train = pd.read_csv(data_train_pathfile)
label_train = pd.read_csv(label_train_pathfile)
data_test = pd.read_csv(data_test_pathfile)
label_test = pd.read_csv(label_test_pathfile)
x_train_tensor = torch.Tensor(data_train.values).float().unsqueeze(2).to(device)
x_test_tensor = torch.Tensor(label_train.values).float().unsqueeze(2).to(device)
y_train_tensor = torch.Tensor(data_test.values).float().to(device)
y_test_tensor = torch.Tensor(label_test.values).float().to(device)
model = torch.load(best_model_pathfile, weights_only=False)

train_predict , test_predict = test(model, scaler= scaler, x_train_tensor=x_train_tensor, x_test_tensor=x_test_tensor)

y_train_actual = scaler.inverse_transform(y_train_tensor.to('cpu').numpy().reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test_tensor.to('cpu').numpy().reshape(-1, 1))

plot_results(y_train_actual, train_predict, y_test_actual, test_predict)
