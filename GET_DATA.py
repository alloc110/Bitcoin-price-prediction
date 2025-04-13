import requests
import datetime
import csv
import time
import os

# Thông tin về Bitcoin và khoảng thời gian
coin_id = "bitcoin"
currency = "usd"
def get_data_by_day():
    days = 365  # Số ngày bạn muốn lấy dữ liệu (1 năm)

    # Tính thời gian bắt đầu (timestamp Unix tính bằng giây)
    now = datetime.datetime.now()
    past = now - datetime.timedelta(days=days)
    timestamp_from = int(past.timestamp())

    # Tính thời gian kết thúc (timestamp Unix tính bằng giây)
    timestamp_to = int(now.timestamp())

    # Gọi API để lấy dữ liệu lịch sử
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency={currency}&from={timestamp_from}&to={timestamp_to}"
    response = requests.get(url)
    output_filename = "data/init/bitcoin_historical_data_.csv"

    # Kiểm tra phản hồi từ API
    if response.status_code == 200:
        data = response.json()

        prices = data.get("prices", [])
        market_caps = data.get("market_caps", [])
        total_volumes = data.get("total_volumes", [])


        with open(output_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Datetime", "Price", "Market Cap", "Volume"])

            for i in range(len(prices)):
                timestamp_ms, price = prices[i]
                _, market_cap = market_caps[i]
                _, volume = total_volumes[i]

                datetime_obj = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
                csv_writer.writerow([datetime_obj, price, market_cap, volume])

        print('Done')

    else:
        print(f"Lỗi khi gọi API: {response.status_code}")
        print(response.text)

def get_data_hourly():
    now_timestamp = int(time.time())
    past_90_days_timestamp = now_timestamp - (90 * 24 * 3600)
    intervals = 'hourly'

    url = url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency={currency}&from={past_90_days_timestamp}&to={now_timestamp}'
    response = requests.get(url)
    output_filename = "data/init/bitcoin_historical_data_by_hourly.csv"

    if response.status_code == 200:
        data = response.json()

        prices = data.get("prices", [])
        market_caps = data.get("market_caps", [])
        total_volumes = data.get("total_volumes", [])

        with open(output_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Datetime", "Price", "Market Cap", "Volume"])

            for i in range(len(prices)):
                timestamp_ms, price = prices[i]
                _, market_cap = market_caps[i]
                _, volume = total_volumes[i]

                datetime_obj = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
                csv_writer.writerow([datetime_obj, price, market_cap, volume])

            print('Done')
    else:
        print(f"Lỗi khi gọi API: {response.status_code}")
        print(response.text)


directory_name = ["init", 'train', 'test']
try:
    for dir in directory_name:
        os.mkdir('data/{0}'.format(dir))
except FileExistsError:
    print(f"Lỗi: Thư mục ' đã tồn tại.")
os.mkdir('tensorboar')
get_data_hourly()
