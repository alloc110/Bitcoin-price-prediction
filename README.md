# Bitcoin Price Prediction

## Table of Contents

* [About The Project](#About-The-Project)
* [Getting Started](#Getting-Started)
  * [Prerequisites](#Prerequisites)
  * [Project structure](#Project-structure)
  * [Installation](#Installation)

## About The Project

## Project structure
```
.
├── dailyModel.py
├── data
│   ├── init
│   │   ├── bitcoin_historical_data_by_hourly.csv
│   │   └── CBBTCUSD.csv
│   ├── test
│   │   ├── data.csv
│   │   ├── label.csv
│   │   ├── test.csv
│   │   └── test_daily.csv
│   └── train
│       ├── data.csv
│       └── label.csv
├── getdata.py
├── linearReg.py
├── LSTM_MODEL.py
├── LSTM_test.py
├── LSTM_train.py
├── models
│   ├── scaler
│   │   ├── scaler1.pkl
│   │   ├── scaler2.pkl
│   │   └── scaler.pkl
│   ├── scaler_daily
│   │   └── scaler.pkl
│   ├── train_model_daily
│   │   ├── best_model_3.pth
│   │   ├── best_model_5.pth
│   │   ├── last_model_3.pth
│   │   └── last_model_5.pth
│   └── train_model_hourly
│       ├── best_model_3.pth
│       ├── best_model_4.pth
│       ├── best_model_5.pth
│       ├── last_model_3.pth
│       ├── last_model_4.pth
│       └── last_model_5.pth
├── __pycache__
│   └── LSTM_MODEL.cpython-312.pyc
├── README.md
├── test_daily.py
└── test.py

```

## Getting Started
### Prerequisites
You need to install the following software on your computer:
* Python 3.9 or latest
* pip
### Installation
1. **Clone the repository:**
<pre>
git clonehttps://github.com/alloc110/Bitcoin-price-prediction
cd Bitcoin-price-prediction
</pre>
2. **Create virtual environment:**
<pre>
python -m venv Bitcoin-price-prediction
</pre>
3. **Install Packages:**
<pre>
pip install requiretment.txt
</pre>

4. **Run test.py:**
<pre>
python test.py
</pre>