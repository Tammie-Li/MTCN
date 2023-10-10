'''
Author: Tammie li
Description: 完成数据装载工作 numpy格式for传统算法 dataloaderfor深度学习模型
FilePath: \data.py
'''
import os
import gc
import copy
import random
import numpy as np
from Utils.preprocess import DataProcess
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class MTCNDataGenerate(Dataset):
    def __init__(self, Dataset, x, y, sub_id, mode):
        # 数据集类型，输入数据，输入标签
        self.dataset = Dataset
        self.x = x
        self.y = y
        field = "train" if mode == True else "test"
        try:
            self.x_mtr = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_mtr_{field}.npy'))
            self.y_mtr = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_mtr_{field}.npy'))
            self.x_msr = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_msr_{field}.npy'))
            self.y_msr = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_msr_{field}.npy'))
        except:
            self.x_mtr, self.y_mtr = self._generate_by_mtr(x)
            self.x_msr, self.y_msr = self._generate_by_msr(x)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_mtr_{field}.npy'), self.x_mtr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_mtr_{field}.npy'), self.y_mtr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'x_msr_{field}.npy'), self.x_msr)
            np.save(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{sub_id:>02d}', f'y_msr_{field}.npy'), self.y_msr)
            print(self.x_mtr.shape, self.y_mtr.shape, self.x_msr.shape, self.y_msr.shape)
            
        
    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        x_mtr = self.x_mtr[9*index: 9*index+9, ...]
        y_mtr = self.y_mtr[9*index: 9*index+9, ...]
        x_msr = self.x_msr[8*index: 8*index+8, ...]
        y_msr = self.y_msr[8*index: 8*index+8, ...]
        return x, y, x_mtr, y_mtr, x_msr, y_msr
    
    def __len__(self):
        return len(self.x)
    
    def _draw(self, data):
        # data->(C, T)
        for i in range(data.shape[0]):
            plt.plot(data[i]+i*0.2)
        plt.show()
    
    def _generate_by_mtr(self, data):
        # 生成 masked temporal recongnition task 数据集
        masked_temporal_martrix = np.array([[0, 28], [29, 34], [35, 40], [41, 51], [52, 64], [65, 85], 
                                            [86, 115], [116, 137], [138, 255]])
        [N, C, T] = data.shape

        # step1：扩充
        expand_data = []
        for i in range(N):
            for j in range(masked_temporal_martrix.shape[0]):
                expand_data.append(data[i])
        expand_data = np.array(expand_data)
        # step2: 修改
        for i in range(N*masked_temporal_martrix.shape[0]):
            sequence = i % masked_temporal_martrix.shape[0]
            for j in range(C):
                raw_signal = expand_data[i, j, masked_temporal_martrix[sequence][0]: masked_temporal_martrix[sequence][1]]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(masked_temporal_martrix[sequence][0], masked_temporal_martrix[sequence][1]):
                    expand_data[i, j, m] = random.gauss(mean, var)
        x_mtr = np.array(expand_data)

        y_mtr = [[i for i in range(masked_temporal_martrix.shape[0])] for i in range(N)]
        y_mtr = np.array(y_mtr).reshape(N*masked_temporal_martrix.shape[0])

        return x_mtr, y_mtr

    def _generate_by_msr(self, data):
        # 生成 masked spatial recongnition task 数据集
        BiosemiRegion = [[0, 1, 2, 3, 4, 5, 6, 62],
                         [7, 8, 9, 10, 11, 16, 17, 18],
                         [13, 15, 20, 22, 24, 29, 31, 33],
                         [36, 59, 38, 42, 43, 45, 47, 60],
                         [37, 39, 41, 44, 46, 48, 61, 63],
                         [50, 51, 52, 53, 54, 55, 56, 57],
                         [12, 14, 19, 21, 23, 28, 30, 32],
                         [25, 26, 27, 34, 35, 42, 49, 58]]
        
        NeuralScanRegion = [[0, 1, 2, 3, 4, 59, 62, 63],
                            [7, 8, 9, 10, 11, 17, 18, 19],
                            [12, 13, 20, 21, 22, 29, 30, 31],
                            [32, 33, 34, 42, 43, 44, 41, 58],
                            [26, 27, 28, 35, 36, 37, 45, 53],
                            [38, 39, 40, 46, 47, 48, 49, 61],
                            [50, 51, 52, 54, 56, 57, 60, 55],
                            [5, 6, 14, 15, 16, 23, 24, 25]]
        region = NeuralScanRegion if self.dataset == "CAS" else BiosemiRegion
        region = np.array(region)

        [N, C, T] = data.shape

        expand_data = []
        # step1: 扩充
        for i in range(N):
            for j in range(region.shape[0]):
                expand_data.append(data[i])
        expand_data = np.array(expand_data)

        # step2: 修改
        for i in range(N*region.shape[0]):
            sequence = i % region.shape[0]
            for j in region[sequence]:
                raw_signal = expand_data[i, j, :]
                mean, var = np.mean(raw_signal), np.var(raw_signal)
                for m in range(T):
                    expand_data[i, j, m] = random.gauss(mean, var)
            # self._draw(expand_data[i])
        x_msr = np.array(expand_data)

        y_msr = [[i for i in range(region.shape[0])] for i in range(N)]
        y_msr = np.array(y_msr).reshape(N*region.shape[0])

        return x_msr, y_msr

class GeneralData(Dataset):
    # 通用模型的数据装载器
    def __init__(self, x, y):
        super(GeneralData, self).__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        return x, y
    
    def __len__(self):
        return len(self.x)


class DataManage:
    def __init__(self, Name, Mode, DataName, SubID, BatchSize):
        # Name (str): the name of method 
        # Mode (bool): True denotes Train mode, False denote Test mode
        # DataName (str): 数据集名
        self.method_name = Name
        self.mode = Mode
        self.dataset = DataName
        self.sub_id = SubID
        self.batch_size = BatchSize
        self.preprocesser = DataProcess()

    def getData(self):
        # 基础网络训练集和测试集数据只有是否shuffle的区别（EEGNet, DeepConvNet et al）
        # 其它有数据增强操作的网络需要自定义（DRL, MTCN et al）

        # 加载原始数据
        field = "train" if self.mode == True else "test"
        x = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{self.sub_id:>02d}', f'x_{field}.npy'))
        y = np.load(os.path.join(os.getcwd(), 'Dataset', self.dataset, f'S{self.sub_id:>02d}', f'y_{field}.npy'))

        x, y = np.array(x, dtype='float32'), np.array(y, dtype='float32')
        # 数据预处理
        x = self.preprocesser.band_pass_filter(data=x, freq_low=0.1, freq_high=40, fs=256)
        x = self.preprocesser.scale_data(x)

        x_npy, y_npy = copy.deepcopy(x), copy.deepcopy(y)

        # （MTCN et al）带来的附加操作
        if self.method_name == "MTCN":
            if self.mode == True:
                data = MTCNDataGenerate(self.dataset, x, y, self.sub_id, self.mode)
                data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
            else:
                data = GeneralData(x, y)
                data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
        else:
            data = GeneralData(x, y)
            data_loader = DataLoader(data, batch_size=self.batch_size, shuffle=(self.mode is True))
        del x, y, field
        gc.collect()
        # data_loader 为深度学习方法准备，后者为传统方法准备
        return data_loader, x_npy, y_npy
            
        

