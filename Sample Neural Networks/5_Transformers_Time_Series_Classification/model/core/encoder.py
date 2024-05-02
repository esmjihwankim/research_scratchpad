from torch import nn

from model.components.positional_encoding import PositionalEncoding
from model.core.encoder_block import EncoderBlock


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 n_head,
                 n_layers,
                 drop_prob,
                 details,
                 device):

        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  details=details,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
