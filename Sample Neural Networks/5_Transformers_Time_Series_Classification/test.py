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
    metrics['correct'] += correct
    total = target.size(0)
    metrics['total'] += total
    print('loss : '
          + str(ce_loss.item())
          + ' correct: '
          + str(((100 * correct)/total))
          + ' target: '
          + str(target.data.cpu().numpy())
          + ' prediction: '
          + str(pred.data.cpu().numpy()))
    return ce_loss


def print_average(metrics):
    loss = metrics['loss']
    print('average loss : ' + str(np.mean(loss)) + 'average correct: ' + str((100 * metrics['correct'])))


def test_model(model, test_loader, device):
    model.eval()
    metrics = dict()
    metrics['loss'] = list()
    metrics['correct'] = 0
    metrics['total'] = 0
    for inputs, labels in test_loader:
        with torch.no_grad():
            inputs = inputs.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)
            pred = model(inputs)
            calc_loss_and_score(pred, labels, metrics)

    print_average(metrics)


batch_size = 10
test_loader = MyTestDataLoader(batch_size=batch_size).get_data_loader()
device = torch.device('cuda')
sequence_len = 187      # sequence length of time series
max_len = 5000          # max time series sequence length
n_head = 2              #
n_layer = 1             # number of encoder layers
drop_prob = 0.1
d_model = 200           # number of dimension for positional encoding
ffn_hidden = 128
feature = 1
model = Transformer(d_model=d_model,
                    n_head=n_head,
                    max_len=max_len,
                    seq_len=sequence_len,
                    ffn_hidden=ffn_hidden,
                    n_layers=n_layer,
                    drop_prob=drop_prob,
                    details=False,
                    device=device).to(device=device)

model.load_state_dict(torch.load('saved_model'))

test_model(device=device, model=model, test_loader=test_loader)

