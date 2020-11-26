import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import code

class Feed_Forward(nn.Module):

    # Defined 2 layer linear network.
    def __init__(self, d_model, hidden_layer=2048, dropout=0.1):
        super(Feed_Forward, self).__init__()
        self.lin1 = nn.Linear(d_model, hidden_layer)
        self.drop1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_layer, d_model)
        self.drop2 = nn.Dropout(dropout)

    # FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    def forward(self, x):
        x = self.drop1(F.relu(self.lin1(x)))
        x = self.drop2(self.lin2(x))
        return x

class Multi_Attention(nn.Module):

    def __new__(cls, d_model,  h=8, dropout=0.1):
        if d_model % h == 0: return super(Multi_Attention, cls).__new__(cls)
        raise AssertionError("Dimension of model not divisible by amount of heads.")

    def __init__(self, d_model, h=8, dropout=0.1):
        super(Multi_Attention, self).__init__()
        
        self.d_model = d_model
        self.d_k = int(d_model / h)
        self.h = h

        self.dropout = nn.Dropout(dropout)
        self.linV = nn.Linear(d_model, d_model)
        self.linK = nn.Linear(d_model, d_model)
        self.linQ = nn.Linear(d_model, d_model)
        self.linW = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.split = lambda t, b, h, d: t.view(b, t.size(1), h, d).transpose(1, 2)
        self.concat = lambda t, b, d: t.transpose(1, 2).contiguous().view(b, -1, d)

    def forward(self, Q, K, V, mask=None):
        
        batch = Q.size(0) # Gets batch size.
        
        # Performs linear operations on Q, K, and V. 
        Q = self.linQ(Q)
        K = self.linK(K)
        V = self.linV(V)

        # Splits Q, K, and V into h heads.
        Q = self.split(Q, batch, self.h, self.d_k) 
        K = self.split(K, batch, self.h, self.d_k) 
        V = self.split(V, batch, self.h, self.d_k)

        atten = self.attention(Q, K, V, mask) # Calculates attention for each head.
        atten = self.concat(atten, batch, self.d_k * self.h) # Concatonates heads.

        out = self.linW(atten) # Applys linear transformation to concatonated outputs.

        return self.drop(out)

    # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
    # Q -> K -> MatMul -> Scale -> Mask -> Softmax -> V -> MatMul
    def attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, V)

class Add_Norm(nn.Module):

    def __init__(self, d_model):
        super(Add_Norm, self).__init__()
        self.norm = nn.LayerNorm(d_model)
    
    # LayerNorm(x + SubLayer(x))
    def forward(self, x, atten):
        return self.norm(torch.add(x, atten))
