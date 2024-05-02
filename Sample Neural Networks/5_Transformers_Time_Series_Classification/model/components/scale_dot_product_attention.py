import math
from torch import nn


class ScaleDotProductAttention(nn.Module):

    def __init__(self, details):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.details = details

    def forward(self, q, k, v, e=1e-12):
        # input is 4 dimension tensor: [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)
        if self.details: print('in Scaled Dot Product, k_t size: ' + str(k_t.size()))
        score = (q @ k_t) / math.sqrt(d_tensor)
        if self.details: print('in Scaled Dot Product, score size: ' + str(score.size()))
        score = self.softmax(score)

        if self.details: print('in Scaled Dot Product, v size: ' + str(v.size()))
        v = score @ v

        if self.details: print('in Scaled Dot Product, v size after matmul:' + str(v.size()))
        return v, score


