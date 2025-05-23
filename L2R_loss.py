import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

EPS = 1e-2
esp = 1e-8

class Fidelity_Loss(torch.nn.Module):

    def __init__(self):
        super(Fidelity_Loss, self).__init__()

    def forward(self, p, g):
        g = g.view(-1, 1)
        p = p.view(-1, 1)
        loss = 1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))

        return torch.mean(loss)

class L2R_Loss(torch.nn.Module):
    '''
    y_pred: [bz]
    y: [bz] -- 需要输入的y是EXIF levels的gt number，不是one-hot labels
    '''

    def __init__(self):
        super(L2R_Loss, self).__init__()

    def prediction_expectation(self, y_pred):
        y_pred = F.softmax(y_pred, dim=1)
        if y_pred.size(1) == 3:
            lvl_expt = y_pred[:, 0] * 1 + y_pred[:, 1] * 2 + y_pred[:, 2] * 3
        elif y_pred.size(1) == 4:
            lvl_expt = y_pred[:, 0] * 1 + y_pred[:, 1] * 2 + y_pred[:, 2] * 3 + y_pred[:, 3] * 4
        else:
            raise NotImplementedError

        return lvl_expt

    def forward(self, y_pred, y):
        # y_pred = self.prediction_expectation(y_pred)

        y_pred = y_pred#.unsqueeze(1)
        y = y.unsqueeze(1)

        preds = y_pred - y_pred.t()
        gts = y - y.t()

        triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
        preds = preds[triu_indices[0], triu_indices[1]]
        gts = gts[triu_indices[0], triu_indices[1]]
        g = 0.5 * (torch.sign(gts) + 1)

        constant = torch.sqrt(torch.Tensor([2.])).to(preds.device) * 0.5
        p = 0.5 * (1 + torch.erf(preds / constant))

        g = g.view(-1, 1)
        p = p.view(-1, 1)

        loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

        return loss
