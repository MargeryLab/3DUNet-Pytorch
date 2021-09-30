
import torch.nn as nn
import torch.nn.functional as F
import torch
from medpy import metric
import numpy as np

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*(self.class_num-1), dtype='float64')
        self.avg = np.asarray([0]*(self.class_num-1), dtype='float64')
        self.sum = np.asarray([0]*(self.class_num-1), dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(torch.argmax(logits, dim=1).cpu().detach().numpy(), targets.cpu().detach().numpy())#[0.99439478 0.47092441 0.52533758]
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(1, 3):
            dice = DiceAverage.calculate_metric_percase(logits == class_index, targets == class_index)
            dices.append(dice)
        return np.asarray(dices)

    @staticmethod
    def calculate_metric_percase(pred, gt):
        pred[pred > 0] = 1
        gt[gt > 0] = 1
        if pred.sum() > 0 and gt.sum() > 0:
            dice = metric.binary.dc(pred, gt)
            # hd95 = metric.binary.hd95(pred, gt)
            return dice
        elif pred.sum() > 0 and gt.sum() == 0:
            return 1
        else:
            return 0