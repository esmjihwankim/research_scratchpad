from torch import nn
from model.components.scale_dot_product_attention import ScaleDotProductAttention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, details):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention(details=details)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
        self.details = details

    def forward(self, q, k, v):
        # 1. perform dot product with predefined weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)



