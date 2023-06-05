import os
from sklearn import metrics
import numpy as np
import pandas as pd
import csv
import torch

from utils import transform_xy, transform_outy

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def predict(net, pre_set, pre_loader, run_time, model_name, log_dir, data_name, device):
    log_dir = os.path.join(log_dir, "logs_test", data_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, model_name+"_pred.csv")
    true_data_path = os.path.join(log_dir, model_name+"_true_data.csv")
    pred_data_path = os.path.join(log_dir, model_name+"_pred_data_"+str(run_time)+".csv")
    net.eval()
    y_true, y_pred = [], []
    for (x, y) in pre_loader:
        x, y = transform_xy(model_name, x , y, device)

        out = net(x)
        out, y = transform_outy(model_name, out, y)
        y_true.extend(y)
        y_pred.extend(out)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = pre_set.inverse_transform(y_true)
    y_pred = pre_set.inverse_transform(y_pred)

    loss_mse = metrics.mean_squared_error(y_true, y_pred)   # MSE 
    loss_mae = metrics.mean_absolute_error(y_true, y_pred)  # MAE
    loss_mape = mape(y_true, y_pred)                        # MAPE
    loss_r2 = metrics.r2_score(y_true, y_pred)              # R2
    print('MSE: {:15.5f}, \tMAE: {:15.5f}, \tMAPE: {:15.5f}, \tR2: {:15.5f}'.format(loss_mse, loss_mae, loss_mape, loss_r2))
    log_list = [loss_mse, loss_mae, loss_mape, loss_r2]

    with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_list)

    if not os.path.exists(true_data_path):
        np.savetxt(true_data_path, y_true, delimiter=',')
    
    np.savetxt(pred_data_path, y_pred, delimiter=',')

    return log_list


def predict_all_dataset(net, model_name, test_set, test_loader, run_time, log_dir, config, device):
    # Make predictions on all datasets
    test_index_list = test_set.index_list
    test_index_list = np.insert(test_index_list, 0, 0)
    data_names = test_set.data_names
    test_batch_size = config['data_params']['test_batch_size']

    simple_x, simple_y = test_set[0]
    test_data_num, input_size = simple_x.shape
    _, output_size = simple_y.shape
    time_step = config['data_params']['time_step']

    net.eval()
    running_loss = 0
    y_true, y_pred = [], []

    # for i, (x, y) in enumerate(test_loader):
        
    #     # 我好傻，真的。
    #     # 我不用写这些玩意，直接在下面y_true和y_pred切片就行了啊...
    #     insert_index = np.searchsorted(test_index_list, i, side='right')  # 返回值的插入位置的右侧索引
    #     if i == 116:
    #         print(1)

    #     if i==0 or (i-test_index_list[insert_index-1])%test_batch_size == 0 or i == test_index_list[insert_index-1]:
    #         x_batch = x
    #         y_batch = y
    #     else:
    #         x_batch = torch.cat((x_batch, x), dim=0)
    #         y_batch = torch.cat((y_batch, y), dim=0)
        
    #     if (i-test_index_list[insert_index-1]+1)%test_batch_size == 0 or i+1 == test_index_list[insert_index-1]:
    #         x_batch = x_batch.transpose(0, 1).to(device)
    #         y_batch = y_batch.transpose(0, 1).to(device)

    #         out = net(x_batch)
    #         y_batch = y_batch[-1,:].cpu().detach().numpy()
    #         out = out[-1,:,0].cpu().detach().numpy()
    #         y_true.extend(y_batch)
    #         y_pred.extend(out)

    # predict by model
    for i, (x, y) in enumerate(test_loader):
        x = x.transpose(0, 1).to(device)
        y = y.transpose(0, 1).to(device)

        out = net(x)
        y = y[-1,:].cpu().detach().numpy()
        out = out[-1,:].cpu().detach().numpy()
        y_true.extend(y)
        y_pred.extend(out)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    y_true_list = []
    y_pred_list = []
    for i in range(len(test_index_list) - 1):  # 按照 test_index_list 切分数据集
        log_folder = os.path.join(log_dir, "logs_test", data_names[i][:-4])
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        log_path = os.path.join(log_folder, model_name+"_pred.csv")
        true_data_path = os.path.join(log_folder, model_name+"_true_data.csv")
        pred_data_path = os.path.join(log_folder, model_name+"_pred_data_"+str(run_time)+".csv")

        start_index = test_index_list[i]
        end_index = test_index_list[i + 1]
        y_true_item = y_true[start_index:end_index]
        y_pred_item = y_pred[start_index:end_index]
        y_true_item = test_set.inverse_transform(y_true_item)
        y_pred_item = test_set.inverse_transform(y_pred_item)
        y_true_list.append(y_true_item)
        y_pred_list.append(y_pred_item)

        loss_mse = metrics.mean_squared_error(y_true_item, y_pred_item)   # MSE 
        loss_mae = metrics.mean_absolute_error(y_true_item, y_pred_item)  # MAE
        loss_mape = mape(y_true_item, y_pred_item)                        # MAPE
        loss_r2 = metrics.r2_score(y_true_item, y_pred_item)              # R2
        print('{:10} --\tMSE: {:15.5f}, \tMAE: {:15.5f}, \tMAPE: {:15.5f}, \tR2: {:15.5f}'.format(data_names[i][:-4], loss_mse, loss_mae, loss_mape, loss_r2))

        log_list = [loss_mse, loss_mae, loss_mape, loss_r2]
        with open(log_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_list)

        if not os.path.exists(true_data_path):
            np.savetxt(true_data_path, y_true_item, delimiter=',')
        
        np.savetxt(pred_data_path, y_pred_item, delimiter=',')

    net.train()

    return running_loss


if __name__ == "__main__":
    import json
    from dataloader import get_dataset_by_folder
    from network.lstm_models import *

    data_path ="./data5"

    # load config
    config_file = open('config.json', 'r').read()
    config = json.loads(config_file)
    device = config['device']
    
    test_set, test_loader = get_dataset_by_folder(data_path, "test", config, shuffle=False)

    model_name = "LSTM_REG"
    input_size = 1
    output_size = 1

    model = eval(model_name+"(input_size, config['network_params']['hidden_size'], output_size, config['network_params']['num_layers'], device=device)")

    predict_all_dataset(model, model_name, test_set, test_loader, 1, "./logs_T", config, device)