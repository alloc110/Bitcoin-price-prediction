import pickle
import sys
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sympy.printing.pretty.pretty_symbology import line_width
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter

from LSTM_train import seq_length

plt.style.use('fivethirtyeight')
seq_length = 3
data_pathfile = 'data/init/CBBTCUSD.csv'
best_model_path = 'train_model_daily/best_model_{0}.pth'.format(seq_length)
last_model_path =  'train_model_daily/last_model_{0}.pth'.format(seq_length)
df = pd.read_csv(data_pathfile, usecols=[1])
df["CBBTCUSD"] = df["CBBTCUSD"].interpolate(method='linear')

scaler = MinMaxScaler(feature_range=(0, 1))
df["CBBTCUSD"]= scaler.fit_transform(df["CBBTCUSD"].values.reshape(-1, 1))

device = "cpu"

def create_sequences(data, seq_length):
    xs , ys = [], []
    for i in range(len(data) - seq_length - 1):
        xs.append(data[i: i+ seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

X, y = create_sequences(df['CBBTCUSD'],  seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


class LSTM (nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

learning_rate = 0.01
num_epochs = 100

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_value = sys.float_info.max
writer = SummaryWriter('tensorboard/experiment_daily_{}'.format(seq_length))  # 'runs/experiment_1' là thư mục log

for epoch in tqdm.tqdm(range(num_epochs)):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor).squeeze()
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if(loss_value > loss.item()):
        torch.save(model, best_model_path)
        loss_value = loss.item()
    torch.save(model, last_model_path)
    writer.add_scalar('training/loss', loss.item(), epoch)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        output = model(X_train_tensor).squeeze()
        val_loss = criterion(output, y_train_tensor)
        total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(y_train)
    writer.add_scalar('validation/loss', avg_val_loss, epoch)

writer.close()

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

# model.eval()
with torch.no_grad():
    train_predict_tensor = model(X_train_tensor).squeeze().to('cpu')
    test_predict_tensor = model(X_test_tensor).squeeze().to('cpu')
train_predict = scaler.inverse_transform(train_predict_tensor.numpy().reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train_tensor.numpy().reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict_tensor.numpy().reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test_tensor.numpy().reshape(-1, 1))

scaler_path= 'scaler_daily/scaler.pkl'

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

plot_results(y_train_actual, train_predict, y_test_actual, test_predict)

print("finish")