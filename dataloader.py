import os
import pandas as pd
import math
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        if not isinstance(data, np.ndarray):
            flat_list = [item for sublist in data for item in sublist]
            data = np.array(flat_list)
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std
    
    def transform_list(self, list_data):
        list_data_tran = []
        for data in list_data:
            mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
            std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
            list_data_tran.append((data - mean) / std)
        return list_data_tran

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    
    
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
    

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='Close', scale=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 10
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path), thousands=",")
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
        df_raw = df_raw[cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)   # df_data.values
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        if self.features=='MS':
            self.data_y = data[border1:border2][:,-1:]
        else:
            self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.pred_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y 
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_by_folder(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', target='Close', scale=True, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 10
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_x_list = []
        self.data_y_list = []
        self.train_data_list = []
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data_names = os.listdir(self.root_path)
        for data_name in self.data_names:
            data_path = os.path.join(self.root_path, data_name)
            df_raw = pd.read_csv(data_path, thousands=",")
            '''
            df_raw.columns: ['Date', ...(other features), target feature]
            '''
            # cols = list(df_raw.columns); 
            if self.cols:
                cols=self.cols.copy()
                cols.remove(self.target)
            else:
                cols = list(df_raw.columns); cols.remove(self.target); cols.remove('Date')
            df_raw = df_raw[['Date']+cols+[self.target]]

            num_train = int(len(df_raw)*0.7)
            num_test = int(len(df_raw)*0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
            border2s = [num_train, num_train+num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
            
            if self.features=='M' or self.features=='MS':
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            elif self.features=='S':
                df_data = df_raw[[self.target]]

            data = df_data.values

            self.data_x_list.append(data[border1:border2].tolist())
            self.train_data_list.append(data[border1s[0]:border2s[0]].tolist())
            if self.features=='MS':
                self.data_y_list.append(data[border1:border2][:,-1:].tolist())
            else:
                self.data_y_list.append(data[border1:border2].tolist())

        self.lengths = [len(lst) - self.seq_len- self.pred_len + 1 for lst in self.data_x_list]
        self.index_list = np.cumsum(self.lengths)
            
        # self.data_x_list = np.array(self.data_x_list)
        # self.data_y_list = np.array(self.data_y_list)

        self.scaler.fit(self.train_data_list)
        self.data_x_list = self.scaler.transform_list(self.data_x_list)
        self.data_y_list = self.scaler.transform_list(self.data_y_list)
        # if self.scale:
        #     train_data = df_data[border1s[0]:border2s[0]]
        #     self.scaler.fit(train_data.values)
        #     # self.scaler.fit(df_data.values)
        #     data = self.scaler.transform(df_data.values)
        # else:
        #     data = df_data.values
    
    def __getitem__(self, index):
        insert_index = np.searchsorted(self.index_list, index, side='right')  # 返回值的插入位置的右侧索引
        data_x = self.data_x_list[insert_index]                               # 获取数据集
        data_y = self.data_y_list[insert_index]

        # 计算获取的位置
        if insert_index == 0:
            s_begin = index
        else:
            s_begin = index-self.index_list[insert_index-1]
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.pred_len
        r_end = s_end + self.pred_len

        seq_x = data_x[s_begin:s_end]
        seq_y = data_y[r_begin:r_end]

        return seq_x, seq_y 
    
    def __len__(self):
        # return len(self.data_x) - self.seq_len- self.pred_len + 1
        return sum(self.lengths)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    



def get_dataset(data_file_path, flag, config):
    data_set = Dataset_Custom(
        root_path=config['data_params']['data_root'],
        data_path=data_file_path,
        size=[config['data_params']['time_step'], config['data_params']['predict_day']], 
        flag=flag,
        features=config['data_params']['features'],
        cols=config['data_params']['cols'],
        target=config['data_params']['target']
    )

    if 'test' == flag or 'val' == flag:
        shuffle=False
    else:
        shuffle=config['data_params']['shuffle']
        
    data_loader = DataLoader(
        data_set,
        batch_size=config['data_params']['test_batch_size'],
        shuffle=shuffle,
        drop_last=False)
    
    return data_set, data_loader


def get_dataset_by_folder(data_folder_path, flag, config, shuffle=True):
    data_set = Dataset_by_folder(
        root_path=data_folder_path,
        size=[config['data_params']['time_step'], config['data_params']['predict_day']], 
        flag=flag,
        features=config['data_params']['features'],
        cols=config['data_params']['cols'],
        target=config['data_params']['target']
    )
    if 'test' == flag or 'val' == flag:
        shuffle=False
    else:
        shuffle=config['data_params']['shuffle']
        
    data_loader = DataLoader(
        data_set,
        batch_size=config['data_params']['test_batch_size'],
        shuffle=shuffle,
        drop_last=False)

    return data_set, data_loader


if __name__ == "__main__":
    flag = 'test'
    data_path = "./data5/"#DIS_2006-01-01_to_2018-01-01.csv"
    data_path2 = "./data1/DIS_2006-01-01_to_2018-01-01.csv"

    config_file = open('config.json', 'r').read()
    config = json.loads(config_file)

    data_set, data_loader = get_dataset_by_folder(data_path, flag, config)

    # data_set, data_loader = get_dataset(data_path2, flag, config)

    print(flag, len(data_set))
    simple_x, simple_y = data_set[0]
    for i, (batch_x, batch_y) in enumerate(data_loader):
        print(batch_x.size())
        print(batch_y.size())
        # break