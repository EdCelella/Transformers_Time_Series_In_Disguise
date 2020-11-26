import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np
from .layers import Encoder, Decoder, Class_Embedding, Positional_Encoder, Traditional_Output
import math
import code

class Transformer(nn.Module):

    def __init__(self, d_model, len_seq, enc_class=None, dec_class=None, \
            output_layer=None, optimiser=None, N=6, h=8, hidden_layer=2048, dropout=0.1, embed_req=True, c=False):
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.embed_req = embed_req

        if embed_req: assert(enc_class is not None and dec_class is not None)

        # Encoder Stack.
        if embed_req: self.enc_embed = Class_Embedding(d_model, enc_class)
        self.enc_pos = Positional_Encoder(d_model, len_seq)
        self.encoder = nn.ModuleList([Encoder(d_model, h, hidden_layer, dropout) for i in range(N)])
        
        # Decoder Stack
        if embed_req: self.dec_embed = Class_Embedding(d_model, dec_class)
        self.dec_pos = Positional_Encoder(d_model, len_seq)
        self.decoder = nn.ModuleList([Decoder(d_model, h, hidden_layer, dropout) for i in range(N)])

        # Output
        if output_layer is None: self.out = Traditional_Output(d_model, dec_class)
        else: self.out = output_layer

        # Initialise parameter weights.
        for i in self.parameters():
            if i.dim() > 1: nn.init.xavier_uniform_(i)

        # Optimiser
        if optimiser is None: self.opt = Optimiser(self)
        else: self.opt = optimiser

        self.c = c

    def forward(self, x, y, inp_mask=None, out_mask=None):
        
        # Encoder pass through.
        enc_input = x
        if self.embed_req: enc_input = self.enc_embed(enc_input)
        enc_input = self.enc_pos(enc_input)
        for i in self.encoder: enc_input = i(enc_input, inp_mask)
        enc_output = enc_input

        # code.interact(local=dict(globals(), **locals()))
        
        # Decoder pass through.
        dec_input = y
        if self.embed_req: dec_input = self.dec_embed(dec_input)
        dec_input = self.dec_pos(dec_input)
        for i in self.decoder: dec_input = i(dec_input, enc_output, inp_mask, out_mask)
        dec_output = dec_input

        # Output.
        return self.out(dec_output)

    def run_epoch(self, data_set, training=True, print_every=500):

        self.train(mode=training)

        output_track=[]
        total_loss = 0
        total_n = 0

        for i in range(len(data_set)):

            if training: self.opt.optimiser.zero_grad()

            # Gets batch and passes through transformer
            enc_input, dec_input, enc_mask, dec_mask, true_vals = data_set[i].enc_inp, data_set[i].dec_inp, data_set[i].enc_mask, data_set[i].dec_mask, data_set[i].true_vals
            output = self.forward(enc_input, dec_input[:, :-1], enc_mask, dec_mask[:, :-1, :-1])

            # calculates loss.
            # loss = self.calc_gradients(output, dec_input[:, 1:], data_set[i].ntokens)
            loss = self.calc_gradients(output, true_vals[:,1:], self.c)
            
            # Updates weights.
            if training == True: self.opt.step()
            
            output_track.append(output)

            n = list(true_vals.size())
            n = n[0] * n[1]

            total_loss += loss
            total_n += n

            # Outputs loss.
            if i % print_every == 0:
                print("Epoch Step: %d Loss: %f Learning Rate: %f" % (i, loss / n, self.opt.rate))

        return total_loss/total_n, output_track

    @staticmethod
    def calc_gradients(predictions, true_vals, c):

        assert predictions.size(1) == true_vals.size(1)
        total = float(0)
        normalize = list(true_vals.size())
        normalize = normalize[0] * normalize[1]
        out_grad = []

        # Calculates loss for each time step.
        for i in range(predictions.size(1)):

            # Gets Batch
            output = ag.Variable(predictions[:, i].data, requires_grad=True)
            # target = ag.Variable(true_vals[:, i]. data, requires_grad=False).double()

            # Calculates loss of batch
            if c == True:
                targets = torch.argmax(true_vals[:,0], 1)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), targets) / normalize
            else:
                loss = F.mse_loss(output.view(-1, output.size(-1)), true_vals[:,i].double()) / normalize

            # Total loss accumulated
            total += loss.data

            # Autograd - Computes gradients
            loss.backward()

            # Appends to list of batches loss
            out_grad.append(output.grad.data.clone())

        # concats and calcs grads.
        out_grad = torch.stack(out_grad, dim=1)
        predictions.backward(gradient=out_grad)
        return total

class Optimiser:

    def __init__(self, model, optimiser=None, scale_factor=1, warmup_steps=4000, beta1=0.9, beta2=0.98, epsilon=1e-9):

        if optimiser is not None: self.optimiser = optimiser
        else: self.optimiser = torch.optim.Adam(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)

        self.scale_factor = scale_factor
        self.warmup_steps = math.pow(warmup_steps, -1.5)
        self.current_step = 0
        self.inv_sqrt_d_model = math.pow(model.d_model, -0.5)

        self.lrate = lambda step: self.inv_sqrt_d_model * min(math.pow(step, -0.5), step * self.warmup_steps)
        self.rate = None

    def step(self):
        self.current_step += 1
        self.rate = self.scale_factor * self.lrate(self.current_step)
        for i in self.optimiser.param_groups: i['lr'] = self.rate
        self.optimiser.step()