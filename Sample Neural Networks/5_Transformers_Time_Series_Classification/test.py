import torch

from torch import nn, optim
from custom_dataset import MyTestDataLoader
import numpy as np

from model.transformer import Transformer

def cross_entropy_loss(pred, target):
    criterion = nn.CrossEntropyLoss()
    loss_class = criterion(pred, target)
    return loss_class

def calc_loss_and_score(pred, target, metrics):
    softmax = nn.Softmax(dim=1)
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    ce_loss = cross_entropy_loss(pred,target)

    metrics['loss'].append(ce_loss.item())
    pred = softmax(pred)
    _, pred = torch.max(pred, dim=1)
    correct = torch.sum(pred == target).item()
    metrics


