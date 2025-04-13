import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm
import LSTM_MODEL
import sys
import csv
import os
from torch.utils.tensorboard import SummaryWriter
import pickle


train_size = 0.8
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
learning_rate = 0.01
num_epochs = 10
seq_length = 3

mode_train = True


device = 'cuda' if torch.cuda.is_available else "cpu"
best_model_pathfile = 'train_model_hourly/best_model_3.pth'
last_model_pathfile = 'train_model_hourly/last_model_3.pth'
data_train_pathfile = 'data/train/data.csv'
label_train_pathfile = 'data/train/label.csv'
data_test_pathfile = 'data/test/data.csv'
label_test_pathfile = 'data/test/label.csv'

# Sliding window data
def create_sequences(data, seq_length):
    xs , ys = [], []
    for i in range(len(data) - seq_length - 1):
        instance = []
        for j in range(seq_length):
            instance.append(data['Price'].loc[i + j])
            instance.append(data['Market Cap'].loc[i + j])
            instance.append(data['Volume'].loc[i + j])

        xs.append(instance)
        ys.append(data['Price'].loc[i+seq_length])

    return np.array(xs), np.array(ys)

def train_lstm(model, x_train, y_train, criterion, optimizer, num_epochs):
    loss_value = sys.float_info.max
    writer = SummaryWriter('tensorboard/experiment_3')  # 'runs/experiment_1' là thư mục log

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(x_train)

        # Calculate the loss
        loss = criterion(output, y_train)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        writer.add_scalar('training/loss', loss.item(), epoch)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            output = model(x_train).squeeze()
            val_loss = criterion(output, y_train)
            total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(y_train)
        writer.add_scalar('validation/loss', avg_val_loss, epoch)

        if loss_value > loss.item():
            torch.save(model, best_model_pathfile)
        torch.save(model, last_model_pathfile)
    return model

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

df = pd.read_csv("data/init/bitcoin_historical_data_by_hourly.csv")

plt.figure(figsize=(15,6))
plt.plot(df.index, df["Price"], label = 'Price', color = 'purple', linewidth = 2)
plt.title("Price of bitcoin by hourly", fontsize = 16)
plt.xlabel('Time', fontsize =14)
plt.ylabel('Price', fontsize = 14)
plt.grid(alpha = 0.3)
plt.legend(fontsize = 13)
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2= MinMaxScaler(feature_range=(0, 1))

df['Price'] = scaler.fit_transform(df['Price'].values.reshape(-1, 1))
df['Market Cap'] = scaler1.fit_transform(df['Market Cap'].values.reshape(-1, 1))
df['Volume'] = scaler2.fit_transform(df['Volume'].values.reshape(-1, 1))
df['Datetime'] = df['Datetime'].str[:10]


x , y = create_sequences(df, seq_length=  seq_length)
# slpit train data and test data
train_size = int(len(x) * train_size)

x1 =  np.random.permutation(x)
y1=  np.random.permutation(y)

x_train, x_test = x1[:train_size] , x1[train_size:]
y_train, y_test = y1[:train_size] , y1[train_size:]

with open(data_train_pathfile, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(x_train)
with open(label_train_pathfile, "w", newline='') as file:
    writer = csv.writer(file)
    for val in y_train:
        writer.writerow([val])

with open(data_test_pathfile, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(x_test)
with open(label_test_pathfile, "w", newline='') as file:
    writer = csv.writer(file)
    for val in y_test:
        writer.writerow([val])


print(x_train.shape)
x_train_tensor = torch.Tensor(x_train).float().unsqueeze(2).to(device)
x_test_tensor = torch.Tensor(x_test).float().unsqueeze(2).to(device)
y_train_tensor = torch.Tensor(y_train).float().to(device)
y_test_tensor = torch.Tensor(y_test).float().to(device)

model = LSTM_MODEL.LSTM_MODEL(input_size, hidden_size, num_layers, output_size).to(device)
criterion = (nn.MSELoss())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if(mode_train == False):
    model = train_lstm(model, x_train_tensor, y_train_tensor, optimizer = optimizer, criterion = criterion  , num_epochs = 1000)
else:
    model = torch.load(best_model_pathfile, weights_only=False)
#
with torch.no_grad():
    train_predict_tensor = model(x_train_tensor).squeeze().to('cpu')
    test_predict_tensor = model(x_test_tensor).squeeze().to('cpu')

scaler_path = 'scaler/scaler.pkl'
scaler_path1= 'scaler/scaler1.pkl'
scaler_path2= 'scaler/scaler2.pkl'

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
with open(scaler_path1, 'wb') as f:
    pickle.dump(scaler1, f)
with open(scaler_path2, 'wb') as f:
    pickle.dump(scaler2, f)


train_predict = scaler.inverse_transform(train_predict_tensor.numpy().reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train_tensor.to('cpu').numpy().reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict_tensor.numpy().reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test_tensor.to('cpu').numpy().reshape(-1, 1))

# plot_results(y_train_actual, train_predict, y_test_actual, test_predict)
