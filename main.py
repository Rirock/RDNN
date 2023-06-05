#coding=utf-8
import argparse
import os
import time
import json
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader 
import pandas as pd
import numpy as np

from dataloader import get_dataset, get_dataset_by_folder
from network.DNM_models import *
from network.lstm_models import *
from train import Train
from predict import predict, predict_all_dataset
from utils import copy_codes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--train_model", default="RDNN", type=str, help="use model")
    parser.add_argument("-d", "--data_path", default="./data", type=str, help="data path")
    parser.add_argument("-n", "--run_times", default=10, type=int, help="run times")
    parser.add_argument("-s", "--start_run_time", default=1, type=int, help="start run times")
    parser.add_argument("--DNM_M", default=20, type=int, help="DNM M")
    parser.add_argument("-l", "--log_name", default="", type=str, help="DNM M")

    args = parser.parse_args()

    model_name = getattr(args, 'train_model')
    data_path = getattr(args, 'data_path')
    run_times = getattr(args, 'run_times')
    start_run_time = getattr(args, 'start_run_time')
    M = getattr(args, 'DNM_M')
    log_name = getattr(args, 'log_name')

    # load config
    config_file = open('config.json', 'r').read()
    config = json.loads(config_file)
    device = config['device']

    # set log
    baseDir = os.path.dirname(os.path.abspath(__name__))
    log_root = os.path.join(baseDir, config['log_params']['log_root']+"_"+model_name+"_"+log_name)
    if not os.path.exists(log_root): os.mkdir(log_root)
    copy_codes(baseDir, log_root)
    
    data_names = os.listdir(data_path)
    for data_name in data_names:
        data_p = os.path.join(data_path, data_name)

        # load dataset
        train_set, train_loader = get_dataset(data_p, "train", config)
        val_set, val_loader = get_dataset(data_p, "val", config)
        test_set, test_loader = get_dataset(data_p, "test", config)

        simple_x, simple_y = train_set[0]
        train_data_num, input_size = simple_x.shape
        _, output_size = simple_y.shape

        print("Dataset: ", data_name)
        print("input_dim:", input_size, "   input_day:", config['data_params']['time_step'], "   predict_day:", input_size, '\n')


        # define network
        if "RDNN" in model_name:
            model = eval(model_name+"(input_size, config['network_params']['hidden_size'], output_size, M, device=device)")
            model = model.to(device)
            model_name_M = model_name+"_M"+str(M)
        elif "REG" in model_name:
            model = eval(model_name+"(input_size, config['network_params']['hidden_size'], output_size, config['network_params']['num_layers'], device=device)")
            model = model.to(device)
            model_name_M = model_name
        elif "MLP" in model_name:
            model = eval(model_name+"(input_size*train_data_num, config['network_params']['hidden_size'], output_size, device=device)")
            model_name_M = model_name
        
        for i in range(start_run_time, run_times+1):
            # training
            print("Training  ---  ", "Time: ", i)
            model.reset_parameters()
            trainer = Train(config, model, train_loader, val_loader, i, model_name_M, log_root, data_name[:-4], device)
            trainer.train()

            # testing
            print("testing  ---  ", "Time: ", i)
            predict(model, test_set, test_loader, i, model_name_M, log_root, data_name[:-4], device)



    # if os.path.isdir(data_path):
    #     data_names = os.listdir(data_path)
    # for data_name_ext in data_names:
    #     data_name = os.path.split(data_name_ext)[-1].split(".")[0]
    #     data_file_path = os.path.join(data_path, data_name_ext)
        
    #     # set log path
    #     log_data_path = os.path.join(log_root, data_name, "logs")
    #     if not os.path.exists(log_data_path): os.makedirs(log_data_path)

    #     # define network
    #     if "RDNN" in model_name:
    #         model = eval(model_name+"(input_size, config['network_params']['hidden_size'], output_size, M, device=device)")
    #         model = model.to(device)
    #         model_name_M = model_name+"_M"+str(M)
    #     elif "REG" in model_name:
    #         model = eval(model_name+"(input_size, config['network_params']['hidden_size'], output_size, config['network_params']['num_layers'], device=device)")
    #         model = model.to(device)
    #         model_name_M = model_name

    #     # start train
    #     for i in range(start_run_time, run_times+1):
    #         print("Train:", data_name, "  Time:", i)
    #         model.reset_parameters()
    #         log_path = os.path.join(log_data_path, model_name_M+'_pred'+'.csv')
    #         trainer = Train(config, model, train_loader, val_loader, i, model_name_M, data_name, log_root, device)
    #         trainer.train()

    #         # predict
    #         predict(model, test_loader, log_path, device)
    #         # torch.save(net, os.path.join(save_path, model_name+"__"+str(i)+".pth"))
