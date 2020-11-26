import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
from .sub_layers import Multi_Attention, Feed_Forward, Add_Norm
import math
import code

class Encoder(nn.Module):

    def __init__(self, d_model, h=8, hidden_layer=2048, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.h = h
        self.hidden_layer = hidden_layer
        self.dropout = dropout

        self.mul_att = Multi_Attention(d_model, h, dropout)
        self.norm1 = Add_Norm(d_model)
        self.ff = Feed_Forward(d_model, hidden_layer, dropout)
        self.norm2 = Add_Norm(d_model)

    def forward(self, x, mask=None): 
        x = self.norm1(x, self.mul_att(x, x, x, mask))
        x = self.norm2(x, self.ff(x))
        return x

class Decoder(nn.Module):

    def __init__(self, d_model, h=8, hidden_layer=2048, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.h = h
        self.hidden_layer = hidden_layer
        self.dropout = dropout

        self.mask_att = Multi_Attention(d_model, h, dropout)
        self.norm1 = Add_Norm(d_model)
        self.mul_att = Multi_Attention(d_model, h, dropout)
        self.norm2 = Add_Norm(d_model)
        self.ff = Feed_Forward(d_model, hidden_layer, dropout)
        self.norm3 = Add_Norm(d_model)

    def forward(self, x, encodings, enc_mask=None, dec_mask=None):
        x = self.norm1(x, self.mask_att(x, x, x, dec_mask))
        x = self.norm2(x, self.mul_att(x, encodings, encodings, enc_mask)) # ONE NEEDS TO BE A DEC_MASK???????
        x = self.norm3(x, self.ff(x))
        return x

class Class_Embedding(nn.Module):

    def __init__(self, d_model, n_class):
        super(Class_Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(n_class, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class Positional_Encoder(nn.Module):

    def __init__(self, d_model, len_seq):
        super(Positional_Encoder, self).__init__()
        self.d_model = d_model

        pos_enc = torch.zeros(len_seq, d_model)
        for i in range(0, len_seq):
            for j in range(0, d_model, 2):
                pos_enc[i, j] = math.sin(i / (math.pow(10000, ((2*j)/d_model))))
                if j+1 < d_model:
                    pos_enc[i, j+1] = math.cos(i / math.pow(10000, ((2*j)/d_model)))
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        len_seq = x.size(1)
        x = x + ag.Variable(self.pos_enc[:,:len_seq], requires_grad=False)
        return x # ADD DROPOUT


class Traditional_Output(nn.Module):

    def __init__(self, d_model, dec_class):
        super(Traditional_Output, self).__init__()
        self.layer = nn.Linear(d_model, dec_class)

    def forward(self, x):
        return F.softmax(self.layer(x), dim=-1)
