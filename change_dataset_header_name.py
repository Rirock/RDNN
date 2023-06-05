import os
import csv
import pandas as pd

old_path = './data4'
new_path = './data5'

data_names = os.listdir(old_path)

for data_name in data_names:
    data_path = os.path.join(old_path, data_name)
    data = pd.read_csv(data_path)

    new_headers = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Change']  # New header list
    data.columns = new_headers

    # Save the renamed data as a new CSV file
    save_path = os.path.join(new_path, data_name)
    data.to_csv(save_path, index=False)