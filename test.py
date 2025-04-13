import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim


device = 'cuda'
data_test_pathfile = 'data/test/test.csv'
scaler_path = 'scaler/scaler.pkl'
scaler_path1 = 'scaler/scaler1.pkl'
scaler_path2 = 'scaler/scaler2.pkl'

best_model_pathfile = 'train_model_hourly/last_model_3.pth'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
with open(scaler_path1, 'rb') as f:
    scaler1 = pickle.load(f)
with open(scaler_path2, 'rb') as f:
    scaler2 = pickle.load(f)

def test(model , scaler,  x_test_tensor ):
    with torch.no_grad():
        test_predict_tensor = model(x_test_tensor).squeeze().to('cpu')
    test_predict = scaler.inverse_transform(test_predict_tensor.numpy().reshape(-1, 1))
    return  test_predict

df = pd.read_csv(data_test_pathfile)
df['Price'] = scaler.fit_transform(df['Price'].values.reshape(-1, 1))
df['Market Cap'] = scaler1.fit_transform(df['Market Cap'].values.reshape(-1, 1))
df['Volume'] = scaler2.fit_transform(df['Volume'].values.reshape(-1, 1))
instance = []
for i in range(3):
    instance.append(df['Price'].loc[i])
    instance.append(df['Market Cap'].loc[i])
    instance.append(df['Volume'].loc[i])

instance = np.array(instance)
instance = instance.reshape((1,9))
x_train_tensor = torch.Tensor(instance).float().unsqueeze(2).to(device)
model = torch.load(best_model_pathfile, weights_only=False)

train_predict = test(model, scaler= scaler,x_test_tensor= x_train_tensor)
print(train_predict)


