'''
Author: Tammie li
Description: 定义各个模型的训练和测试任务（通用深度学习网络可以共用任务）
FilePath: \task.py
'''

import torch
import pickle
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import torch.nn as nn
import sklearn.decomposition
import sklearn.discriminant_analysis
import pywt
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression

from Manage.loss import LossFunction


class MTCNTrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = LossFunction()
        self.dataset = dataset


    def train(self, dataloader, epochs):
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for epoch in range(epochs):
                self.model.train()
                running_loss = 0.0
                correct_num, correct_vto_num, correct_msp_num = 0, 0, 0
                sum_num = 0
                for index, data in enumerate(dataloader):
                    # generate new data and labels

                    x_train_main_task, y_train_main_task, x_train_vto_task, y_train_vto_task, x_train_msp_task, y_train_msp_task = data
                    batch_size = x_train_main_task.shape[0] if index == 0 else batch_size

                    sum_num += x_train_main_task.shape[0]
                    x_train_main_task = x_train_main_task.type(torch.FloatTensor)
                    x_train_vto_task = x_train_vto_task.type(torch.FloatTensor)
                    x_train_msp_task = x_train_msp_task.type(torch.FloatTensor)

                    x_train_main_task, y_train_main_task = x_train_main_task.to(self.device), y_train_main_task.to(self.device)
                    x_train_vto_task, y_train_vto_task = x_train_vto_task.to(self.device), y_train_vto_task.to(self.device)
                    x_train_msp_task, y_train_msp_task = x_train_msp_task.to(self.device), y_train_msp_task.to(self.device)

                    x_train_vto_task = x_train_vto_task.reshape(x_train_vto_task.shape[0]*x_train_vto_task.shape[1], x_train_vto_task.shape[2], x_train_vto_task.shape[3])
                    x_train_msp_task = x_train_msp_task.reshape(x_train_msp_task.shape[0]*x_train_msp_task.shape[1], x_train_msp_task.shape[2], x_train_msp_task.shape[3])
                    y_train_vto_task = y_train_vto_task.reshape(y_train_vto_task.shape[0]*y_train_vto_task.shape[1])
                    y_train_msp_task = y_train_msp_task.reshape(y_train_msp_task.shape[0]*y_train_msp_task.shape[1])

                    pred_primary, loss_main = self.model(x_train_main_task, "main")
                    pred_vto, loss_vto = self.model(x_train_vto_task, "vto")
                    pred_msp, loss_msp = self.model(x_train_msp_task, "msp")

                    loss = self.criterion.calculateTrainStageLoss(pred_primary, y_train_main_task, pred_vto, y_train_vto_task,
                                                                pred_msp, y_train_msp_task)
                    
                    loss += loss_main + loss_vto + loss_msp

                    _, pred = torch.max(pred_primary, 1)
                    _, pred_vto = torch.max(pred_vto, 1)
                    _, pred_msp = torch.max(pred_msp, 1)
                    
                    for i in range(y_train_main_task.shape[0]):
                        if y_train_main_task[i] == pred[i]:
                            correct_num += 1
                    for i in range(y_train_vto_task.shape[0]):
                        if y_train_vto_task[i] == pred_vto[i]:
                            correct_vto_num += 1
                    for i in range(y_train_msp_task.shape[0]):
                        if y_train_msp_task[i] == pred_msp[i]:
                            correct_msp_num += 1
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    running_loss += float(loss.item())

                _loss = running_loss / sum_num
                tmp_acc = correct_num / sum_num * 100
                tmp_acc_vto = correct_vto_num / sum_num / 9 * 100
                tmp_acc_msp = correct_msp_num / sum_num / 8 * 100

                # print(f'Train loss: {_loss:.4f}\tTrain acc: {acc:.2f}%')
                pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                pbar.set_postfix(loss = _loss, acc = tmp_acc, acc_vto=tmp_acc_vto, acc_msp=tmp_acc_msp)
                pbar.update(1)
            path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
            torch.save(self.model.state_dict(), path)


class MTCNTestTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.dataset = dataset

    def test(self, data):
        self.model.load_state_dict(torch.load(f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'))
        running_loss = 0.0
        correct_num = 0
        self.model.eval()
        batch_size = None
        preds = []
        ys = []
        pred_score =[]
        test_num = 0
        # fea = []
        for index, data in enumerate(data):
            x, y = data
            batch_size = x.shape[0] if index == 0 else batch_size
            x = torch.tensor(x).to(torch.float32)
            y = torch.tensor(y).to(torch.long)  
            x, y = x.to(self.device), y.to(self.device)
            y_pred, loss = self.model(x, "main")
            _, pred = torch.max(y_pred, 1)

            correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
            preds.extend(pred.cpu().numpy().tolist())
            ys.extend(y.cpu().tolist())
            pred_score.extend(y_pred.cpu().detach().numpy())  
            test_num += x.shape[0]

        batch_num = test_num // batch_size
        acc = correct_num / test_num * 100
        print(f'Test acc: {acc:.2f}%')   

        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y', ys)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_preds', preds)
        np.save(f'PredictionResult/{self.dataset}_{self.model.__class__.__name__}_S{self.sub_id:>02d}_y_pred', np.array(pred_score))

class CPTrainTask:
    def __init__(self, SubID, MODEL, DEVICE, dataset):
        self.sub_id = SubID
        self.device = DEVICE
        self.model = MODEL
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = dataset
    
    def train(self, dataloader, epochs):
        print("Stage 1: representation learning")
        with tqdm(total=epochs, desc='Epoch', leave=True, ncols=100, unit_scale=True) as pbar:
            for mode in ["SSL", "SL"]: 
                for epoch in range(epochs):
                    self.model.train()
                    running_loss = 0.0
                    correct_num, correct_vto_num, correct_msp_num = 0, 0, 0
                    sum_num = 0
                    for index, data in enumerate(dataloader):
                        x, y = data
                        # 数据变换
                        if mode == "SSL":
                            value = x.shape[0] // 2
                            x[value: , :, :] = torch.cat((x[:value, :, :128], x[value: , :, 128:]), dim=2)
                            y[: value], y[value: ] = 1, 0
                        batch_size = x.shape[0] if index == 0 else batch_size
                        x = torch.tensor(x).to(torch.float32)
                        y = torch.tensor(y).to(torch.long)
                        x, y = x.to(self.device), y.to(self.device)   
                        if mode == "SSL":
                            y_pred = self.model(x, "feature")
                        else:
                            y_pred = self.model(x, "classifier")
                        loss = self.criterion(y_pred, y)
                        _, pred = torch.max(y_pred, 1)
                        correct_num += np.sum(pred.cpu().numpy() == y.cpu().numpy())
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        running_loss += float(loss.item())
                        sum_num += x.shape[0]
                    batch_num = sum_num // batch_size
                    _loss = running_loss / (batch_num + 1)
                    acc = correct_num / sum_num * 100
                    pbar.update(1)
                    pbar.set_description(f'Epoch[{epoch}/{epochs}]')
                    pbar.set_postfix(loss = _loss, acc = acc)
        path = f'Checkpoint/{self.dataset}_{self.model.__class__.__name__}_{self.sub_id:>02d}.pth'
        torch.save(self.model.state_dict(), path)        



class TaskManage:
    def __init__(self, SubjectID, Name, Type, Epoch, Model, Data, DEVICE, DatasetName):
        self.model_name = Name
        self.task_type = Type
        self.epoch = Epoch
        self.model = Model
        self.data = Data
        self.device = DEVICE
        self.sub_id = SubjectID
        self.dataset_name = DatasetName

    def goTask(self):
        if self.model_name == "MTCN":
            if(self.task_type == True):
                general = MTCNTrainTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.train(self.data, self.epoch)
            else:
                general = MTCNTestTask(self.sub_id, self.model, self.device, self.dataset_name)
                general.test(self.data)





