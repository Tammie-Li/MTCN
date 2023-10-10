'''
Author: Tammie li
Description: Define model
FilePath: \model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MTCN(nn.Module):
    def __init__(self, n_class_primary, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32):
        super(MTCN, self).__init__()

        self.n_class_primary = n_class_primary
        self.channels = channels
        self.n_kernel_t = n_kernel_t
        self.n_kernel_s = n_kernel_s
        self.dropout = dropout
        self.kernel_length = kernel_length

        
        self.block_shared_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_main_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_mtr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )
        self.block_specific_msr_feature_extractor = nn.Sequential(
            # 原block1
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, 8, (1, 32), bias=False),
            nn.BatchNorm2d(8),
            # 原block2
            nn.Conv2d(8, 16, (64, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout)
        )

        self.block_feature_fusion = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//8-1, self.kernel_length//8, 0, 0)),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, self.kernel_length//4), groups=self.n_kernel_s, bias=False),
            nn.Conv2d(self.n_kernel_s, self.n_kernel_s, (1, 1), bias=False),
            nn.BatchNorm2d(self.n_kernel_s),
            nn.ELU()
        )
        self.main_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.vto_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )
        self.msp_task_projection_head =  nn.Sequential(
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropout)
        )

        # Fully-connected layer
        self.primary_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, self.n_class_primary)
        )
        self.vto_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 9)
        )
        self.msp_task_classifier = nn.Sequential(
            nn.Linear(self.n_kernel_s*T//32, 8)
        )

    def calculate_orthogonal_constraint(self, feature_1, feature_2):
        assert feature_1.shape == feature_2.shape, "the dimension of two matrix is not equal"
        N, C, H, W = feature_1.shape
        feature_1, feature_2 = torch.reshape(feature_1, (N*C, H, W)), torch.reshape(feature_2, (N*C, H, W))
        weight_squared = torch.bmm(feature_1, feature_2.permute(0, 2, 1))
        # weight_squared = torch.norm(weight_squared, p=2)
        ones = torch.ones(N*C, H, H, dtype=torch.float32).to(torch.device('cuda:0'))
        diag = torch.eye(H, dtype=torch.float32).to(torch.device('cuda:0'))

        loss = ((weight_squared * (ones - diag)) ** 2).sum()
        return loss

    def forward(self, x, task_name):
        '''
        @description: Complete the corresponding task according to the task tag
        '''
        # extract features
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        
        fea_shared_extract = self.block_shared_feature_extractor(x)
        fea_after_fusion = self.block_feature_fusion(fea_shared_extract)

        if task_name == "main":
            # 推理过程
            fea_specific_main = self.block_specific_main_feature_extractor(x)
            fea_main = fea_specific_main + fea_after_fusion
            fea_main = self.main_task_projection_head(fea_main)
            fea_main = fea_main.view(fea_main.size(0), -1)
            logits_main = self.primary_task_classifier(fea_main)
            pred_main = F.softmax(logits_main, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_main, fea_shared_extract)

            return pred_main, orthogonal_constraint

        elif task_name == "vto":
            # 推理过程
            fea_specific_vto = self.block_specific_mtr_feature_extractor(x)
            fea_vto = (fea_specific_vto + fea_after_fusion)
            fea_vto = self.vto_task_projection_head(fea_vto)
            fea_vto = fea_vto.view(fea_vto.size(0), -1)
            logits_vto = self.vto_task_classifier(fea_vto)
            pred_vto = F.softmax(logits_vto, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_vto, fea_shared_extract)

            return pred_vto, orthogonal_constraint

        elif task_name == "msp":
            # 推理过程
            fea_specific_msp = self.block_specific_msr_feature_extractor(x)
            fea_msp = (fea_specific_msp + fea_after_fusion)
            fea_msp = self.msp_task_projection_head(fea_msp)
            fea_msp = fea_msp.view(fea_msp.size(0), -1)
            logits_msp = self.msp_task_classifier(fea_msp)
            pred_msp = F.softmax(logits_msp, dim = 1)
            # 损失计算
            orthogonal_constraint = self.calculate_orthogonal_constraint(fea_specific_msp, fea_shared_extract)

            return pred_msp, orthogonal_constraint
        else:
            assert("TaskName Error!")



if __name__ == "__main__":
    data = torch.tensor(np.random.rand(64, 64, 256)).to(torch.float32)
    model = MTCN(n_class_primary=2, T = 256, channels=64, n_kernel_t=8, n_kernel_s=16, dropout=0.5, kernel_length=32)

    a = model(data, "main")

