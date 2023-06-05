#coding=utf-8
import torch
import numpy as np
import pandas as pd
import time
import os


from utils import format_runtime, transform_xy

class Train:
    '''
    The Train class sets up train process, combining DataLoader, network, loss, and other modules
    Inputs: config, logger, net, train_data_loader, val_data_loader, device
        - config: config file for init Train class
        - net: the network to train
        - train_loader: the class<DataLoader>, to load train data
        - val_loader: the class<DataLoader>, to load valid data
        - run_time: the first few times of training
        - device: the device will run
    '''
    def __init__(self, config, net, train_loader, val_loader, run_time, model_name, log_root, data_name, device):
        self.net = net
        self.model_name = model_name
        self.data_name = data_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_time = run_time
        self.log_root = log_root
        self.device = device

        self.lr = config['train_params']['lr']
        self.loss = eval('torch.nn.'+config['train_params']['loss']+'Loss()')
        self.opt = config['train_params']['optimizer']
        
        self.max_epoch = config['train_params']['epoch']
        self.show_steps = config['train_params']['show_steps']
        self.save_mode = config['train_params']['save_mode']

        self.stop_epoch = config['train_params']['early_stop_epoch']
        self.update_lr_epoch = config['train_params']['update_lr_epoch']
        self.no_improve = 0                              # how many epochs have not been improved
        self.best_val_loss = float('inf')                # current best val loss
        self.weight_save_path = os.path.join(log_root, "weight", data_name, self.model_name+'_best_'+str(run_time)+'.pth')
        self.log_save_path = os.path.join(log_root, "logs_train", data_name, self.model_name+'_Train_'+str(run_time)+'.csv')
        self.mkdir()

    def mkdir(self):
        weight_save_path, _ = os.path.split(self.weight_save_path)
        log_save_path , _ = os.path.split(self.log_save_path)
        if not os.path.exists(weight_save_path):
            os.makedirs(weight_save_path)
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)

    def set_opt(self, **kwargs):
        '''
        Set optimizer by kwargs. Because the params of each optimizer are quite different, therefore this function set
        optimizer by kwargs<dict>. If parameter is None, set default optimizer via self.lr.
        '''
        if kwargs == {}:
            self.opt = eval('torch.optim.'+self.opt)(self.net.parameters(), lr=self.lr)
        else:
            self.opt = eval('torch.optim.'+self.opt)(self.net.parameters(), kwargs)    

    def loss_fc(self, x, y):
        if "RDNN" in self.model_name or "REG" in self.model_name:
            loss = self.loss(x.cpu(), y.cpu())
        elif "MLP" in self.model_name:
            loss = self.loss(x.cpu(), y[:, -1:].cpu())
        return loss

    def train(self):
        self.net.train()
        self.set_opt()
        log_list =[]
        step_time = time.time()
        num_batches = len(self.train_loader)
        
        for epoch in range(self.max_epoch):
            train_loss = 0
            self.opt.zero_grad()
            for (x, y) in self.train_loader:
                x, y = transform_xy(self.model_name, x , y, self.device)

                out = self.net(x)
                loss=self.loss_fc(out, y)
                train_loss += loss.item()

                loss.backward()
                self.opt.step()
            train_loss /= num_batches

            # valid
            val_loss = self.valid()
            if (epoch + 1) % 100 == 0:  # 每 100 次输出结果
                print('Epoch: {}, Loss: {:.5f} Time: {}, LR: {}'.format(epoch + 1, train_loss, format_runtime(time.time()-step_time), self.opt.param_groups[0]['lr']))
                print('Valid : Loss: {}, BestLoss: {},'.format(self.best_val_loss, self.best_val_loss))
                step_time = time.time()

            if val_loss < self.best_val_loss:            # save model
                self.best_val_loss = val_loss
                self.save_model(self.net, mode=self.save_mode)
                self.no_improve = 0
            else:
                self.no_improve += 1

            if self.no_improve == self.stop_epoch:       # early stop
                break
            if (self.no_improve+1) % (self.update_lr_epoch+1) == 0:  # update_learning_rate
                lr = self.opt.param_groups[0]['lr']
                for param in self.opt.param_groups:
                    param['lr'] = lr * 0.9
            
            log_list.append([epoch+1, train_loss, val_loss, self.opt.param_groups[0]['lr']])
        
        final_weight_save_path = os.path.join(self.log_root, "weight", self.data_name, self.model_name+'_last_'+str(self.run_time)+'.pth')
        torch.save(self.net, final_weight_save_path)
        save_log = pd.DataFrame(log_list)
        save_log.to_csv(self.log_save_path,mode='w',header=False,index=False)
        self.net = torch.load(self.weight_save_path)

    def valid(self):
        self.net.eval()
        running_loss = 0
        num_batches = len(self.val_loader)
        for (x, y) in self.val_loader:
            x, y = transform_xy(self.model_name, x , y, self.device)

            out = self.net(x)
            loss=self.loss_fc(out, y)
            running_loss += loss.item()
        self.net.train()

        running_loss /= num_batches

        return running_loss
    

    def save_model(self, net, mode='min'):
        if mode == 'min':
            savepath = self.weight_save_path
            torch.save(net, savepath)
        elif mode == 'max':
            savepath = self.weight_save_path
            torch.save(net, savepath)
        else:
            raise ValueError('Save mode must be in ["max", "min"], error {}'.format(mode))