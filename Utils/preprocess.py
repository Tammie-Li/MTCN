
# Author: Tammie li
# Description: 定义预处理方法 预处理方法是脑电信号处理的关键一环 必要时可以将其plot出来
# FilePath: \Utils\preprocess.py


from sklearn import preprocessing
from scipy import signal
import numpy as np
from scipy.linalg import sqrtm, inv
from collections import defaultdict


class DataProcess:
    def __init__(self):
        pass
    
    def scale_data(self, data):
        # 归一化数据 data = [N, C, T]
        scaler = preprocessing.StandardScaler()
        for i in range(data.shape[0]):
            data[i, :, :] = scaler.fit_transform(data[i, :, :])
        return data

    def band_pass_filter(self, data, freq_low, freq_high, fs):
        # 带通滤波
        wn = [freq_low * 2 / fs, freq_high * 2 / fs]
        b, a = signal.butter(3, wn, "bandpass")
        for trial in range(data.shape[0]):
            data[trial, ...] = signal.filtfilt(b, a, data[trial, ...], axis=1)
        return data

    def euclidean_space_alignment(self, data):
        """Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach"""
        # data->(N, C, T), 需要先执行滤波操作
        # 公式10-计算协方差
        r = 0
        for trial in data:
            cov = np.cov(trial, rowvar=True)
            r += cov
        r = r/data.shape[0]
        # 公式11
        r_op = inv(sqrtm(r))

        results = np.matmul(r_op, data)
        return results

if __name__ == "__main__":
    data = np.random.rand(10, 64, 256)
    prePocessor = DataProcess()
    result = prePocessor.euclidean_space_alignment(data)



