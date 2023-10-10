'''
Author: Tammie li
Description: Define loss function
FilePath: \loss.py
'''
import torch
import torch.nn as nn

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        length = len(x)-1
        for i, loss in enumerate(x):
            if i == length:
                loss_sum += 1 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
            else:
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i])
        return loss_sum


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.loss = AutomaticWeightedLoss(3)
    
    
    def calculateTrainStageLoss(self, pred_primary, label_primary, pred_vto, label_vto, pred_msp, label_msp):
        
        loss_primary = self.criterion(pred_primary, label_primary.type(torch.cuda.LongTensor))
        loss_vto = self.criterion(pred_vto, label_vto.type(torch.cuda.LongTensor))
        loss_msp = self.criterion(pred_msp, label_msp.type(torch.cuda.LongTensor))

        loss = self.loss(loss_primary, loss_vto, loss_msp)
        
        return loss

    def calculateTestStageILoss(self, pred_vto, label_vto, pred_msp, label_msp):
        loss_vto = self.criterion(pred_vto, label_vto.type(torch.cuda.LongTensor))
        loss_msp = self.criterion(pred_msp, label_msp.type(torch.cuda.LongTensor))
        
        loss = self.loss(loss_vto, loss_msp)
        
        return loss

    def calculateTestStageIILoss(self, pred_primary, label_primary):
        loss_primary = self.criterion(pred_primary, label_primary.type(torch.cuda.LongTensor))
        
        return loss_primary