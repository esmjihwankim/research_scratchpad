import torch

import copy
from torch import nn, optim
from torchinfo import summary
from model.transformer import Transformer
from custom_dataset import MyDataLoader
import numpy as np


def cross_entropy_loss(pred, target):
    criterion = nn.CrossEntropyLoss()
    loss_class = criterion(pred, target)
    return loss_class


def calc_loss_and_score(pred, target, metrics):
    softmax = nn.Softmax(dim=1)
    pred = pred.squeeze(-1)
    target = target.squeeze(-1)
    ce_loss = cross_entropy_loss(pred, target)
    metrics['loss'].append(ce_loss.item())
    pred = softmax(pred)

    _, pred = torch.max(pred, dim=1)
    metrics['correct'] += torch.sum(pred == target).item()
    metrics['total'] += target.size(0)

    return ce_loss


def print_metrics(main_metrics_train, main_metrics_val, metrics, phase):
    correct = metrics['correct']
    total = metrics['total']
    accuracy = 100 * correct / total
    loss = metrics['loss']
    if phase == 'train':
        main_metrics_train['loss'].append(np.mean(loss))
        main_metrics_train['accuracy'].append(accuracy)
    else:
        main_metrics_val['loss'].append(np.mean(loss))
        main_metrics_val['accuracy'].append(accuracy)
    result = "phase: " + str(phase) \
    + '\nloss: {:4f}'.format(np.mean(loss)) + 'accuracy : {:4f}'.format(accuracy)
    return result


def train_model(dataloaders, model, optimizer, num_epochs=100):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_dict = dict()
    train_dict['loss'] = list()
    train_dict['accuracy'] = list()
    val_dict = dict()
    val_dict['loss'] = list()
    val_dict['accuracy'] = list()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = dict()
            metrics['loss'] = list()
            metrics['correct'] = 0
            metrics['total'] = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)
                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss_and_score(outputs, labels, metrics)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            print(print_metrics(main_metrics_train=train_dict,
                                main_metrics_val=val_dict,
                                metrics=metrics,
                                phase=phase))
            epoch_loss = np.mean(metrics['loss'])

            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss

    print('Best val loss: {:4f}'.format(best_loss))


device = torch.device('cuda')
sequence_len = 187
max_len = 5000
n_head = 2
n_layer = 1
drop_prob = 0.1
d_model = 200
ffn_hidden = 128
feature = 1
batch_size = 100
model = Transformer(d_model=d_model,
                    n_head=n_head,
                    max_len=max_len,
                    seq_len=sequence_len,
                    ffn_hidden=ffn_hidden,
                    n_layers=n_layer,
                    drop_prob=drop_prob,
                    details=False,
                    device=device).to(device=device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloaders = MyDataLoader(batch_size=batch_size).get_data_loader()

train_model(dataloaders=dataloaders,
            model=model,
            optimizer=optimizer,
            num_epochs=20)

torch.save(model.state_dict(), 'saved_model')

