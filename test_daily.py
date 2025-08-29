import pickle
import pandas as pd
import torch
import numpy as np

data_test_pathfile =  'data/test/test_daily.csv'
df = pd.read_csv(data_test_pathfile)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_model_path = 'models/train_model_daily/best_model_5.pth'
model = torch.load(best_model_path, weights_only= False)

scaler_path= 'models/scaler_daily/scaler.pkl'
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

df["CBBTCUSD"]= scaler.fit_transform(df["CBBTCUSD"].values.reshape(-1, 1))
lt = []
for i in df['CBBTCUSD']:
    lt.append(i)
lt = np.array(lt).reshape((1,5))
X_train_tensor = torch.tensor(lt, dtype=torch.float32).unsqueeze(2).to(device)
with torch.no_grad():
    train_predict_tensor = model(X_train_tensor).squeeze().to('cpu')

train_predict = scaler.inverse_transform(train_predict_tensor.numpy().reshape(-1, 1))
print(train_predict)