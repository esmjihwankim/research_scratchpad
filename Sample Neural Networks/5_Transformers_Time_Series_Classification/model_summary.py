import torch

from torchinfo import summary
from model.transformer import Transformer

device = torch.device('cuda')
sequence_len = 187
max_len = 5000
n_head = 4
n_layer = 2
drop_prob = 0.1
d_model = 200
ffn_hidden = 512
feature = 1
model = Transformer(d_model=d_model,
                    details=True,
                    n_head=n_head,
                    max_len=max_len,
                    seq_len=sequence_len,
                    ffn_hidden=ffn_hidden,
                    n_layers=n_layer,
                    drop_prob=drop_prob,
                    device=device)
batch_size = 7
summary(model, input_size=(batch_size, sequence_len, feature), device=device)
print(model)

