
from torch import nn

from model.components.layer_normalization import LayerNorm
from model.components.multi_head_attention import MultiHeadAttention
from model.components.position_wise_feedforward import PositionwiseFeedForward


class EncoderBlock(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob, details):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head, details=details)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.details = details
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x)
        if self.details: print('in encoder layer: '+str(x.size()))
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        if self.details: print('in encoder after norm layer: ' + str(x.size()))
        # 3. position-wise feed forward network
        _x = x
        x = self.ffn(x)
        if self.details: print('in encoder after ffn: ' + str(x.size()))
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

