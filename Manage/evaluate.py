# Author: Tammie li
# Description: Define the evaluation metrics
# FilePath: \DRL\Utils\evaluate.py
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import copy


class EvaluateMetric:
    def __init__(self) -> None:
        self.KAPPA = None
        self.AUC = None
        self.BA = None
        self.TPR = None
        self.FPR = None
        self.F1 = None
        self.ACC = None
        self.P = None


class EvaluateManage:
    def __init__(self, subject_id, dataset_name, model_name):
        self.subject_id = subject_id
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.pred = np.load(f'PredictionResult/{self.dataset_name}_{self.model_name}_S{self.subject_id:>02d}_preds.npy')
        self.y = np.load(f'PredictionResult/{self.dataset_name}_{self.model_name}_S{self.subject_id:>02d}_y.npy')
        self.y_pred = np.load(f'PredictionResult/{self.dataset_name}_{self.model_name}_S{self.subject_id:>02d}_y_pred.npy')

        self.score = EvaluateMetric()
    
    def calculate_metric_score(self):
        # 计算
        sum_num_non_tar = 0
        cor_num_non_tar = 0
        sum_num_tar = 0
        cor_num_tar = 0

        for idx, label in enumerate(self.y):
            if label == 0:
                if self.pred[idx] == label:
                    cor_num_non_tar += 1
                sum_num_non_tar += 1
            elif label == 1:
                if self.pred[idx] == label:
                    cor_num_tar += 1
                sum_num_tar += 1

        print(cor_num_tar, cor_num_non_tar, sum_num_tar, sum_num_non_tar)


        TP = cor_num_tar
        TN = cor_num_non_tar
        FP = sum_num_non_tar - cor_num_non_tar
        FN = sum_num_tar - cor_num_tar

        # print(TP, TN, FP, FN)

        ACC = round((TP + TN) / (TP + TN + FP + FN), 4)
        BA = round((TP / (TP + FN) + TN / (TN + FP)) / 2, 4)
        TPR = round(TP / (TP + FN), 4)
        FPR = round(FP / (TN + FP), 4)
        P = round(TP / (TP + FP), 4)
        F1 = round(2 * P * TPR / (P + TPR), 4)
        P_o, P_e = (TP + TN) / (TP + TN + FP + FN), (cor_num_tar**2 + cor_num_non_tar**2) / (sum_num_tar + sum_num_non_tar)**2
        KAPPA = (P_o - P_e) / (1 - P_e)

        precision, recall, _thresholds = precision_recall_curve(self.y, self.y_pred[:, 1])
        AUC = auc(recall, precision)

        AUC = roc_auc_score(self.y, self.y_pred[:, 1])
        
        self.score.ACC, self.score.BA, self.score.TPR, self.score.FPR, self.score.P, self.score.F1, \
        self.score.KAPPA, self.score.AUC = ACC, BA , TPR, FPR, P , F1, KAPPA, AUC
        
        return self.score

    














