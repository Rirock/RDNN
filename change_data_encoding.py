# Change the file encoding format to utf-8
import pandas as pd
import os
import chardet

def get_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())  # or readline if the file is large

    #print(result['encoding'])
    return result['encoding']

path = './data1/'
new_path = './data7'
if not os.path.exists(new_path):
    os.mkdir(new_path)

dataset_names = os.listdir(path)
for dataset_name in dataset_names:
    dataset_path = os.path.join(path, dataset_name)
    if dataset_path.split('.')[-1] == 'csv': 
        encoding = get_encoding(dataset_path)
        init_data = pd.read_csv(dataset_path, encoding = encoding)
        new_dataset_path = os.path.join(new_path, dataset_name)
        init_data.to_csv(new_dataset_path, encoding='utf-8')
            
    print(dataset_name, encoding, " -> UTF-8    ---Done!")