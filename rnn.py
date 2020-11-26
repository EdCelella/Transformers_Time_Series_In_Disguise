import torch
from torch import nn
from Transformer.transformer import Optimiser
import torch.nn.functional as F
import torch.autograd as ag

class RNN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim=2048, layers=4, dropout=0.1, c=False):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.layers = layers
        self.d_model = inp_dim

        # RNN Layer
        self.rnn = nn.RNN(int(inp_dim), int(hidden_dim), int(layers), batch_first=bool(True), dropout=float(dropout))   
        self.out = nn.Linear(hidden_dim, out_dim)

        self.opt = Optimiser(self)

        self.c = c

        self.is_training = True
    
    def forward(self, x):

        output, hidden = self.rnn(x, None)
        
        # output = output.contiguous().view(-1, self.hidden_dim)

        if self.c == False:
            output = self.out(output)
        else:
            output = F.softmax(self.layer(x), dim=-1)
        
        return output

    def run_epoch(self, data_set, training=True, print_every=500, device='cpu'):

        if training == True:
            self.train()
        else:
            self.eval()

        self.is_training = training

        output_track=[]
        total_loss = 0
        total_n = 0

        for i in range(len(data_set)):

            if training: self.opt.optimiser.zero_grad()

            # Gets batch and passes through transformer
            inp, target = data_set[i].enc_inp, data_set[i].true_vals
            output = self.forward(inp)

            if training == True:
                self.train()
            else:
                self.eval()

            # calculates loss.
            loss = self.calc_gradients(output, target, self.c)
            
            # Updates weights.
            if training == True: self.opt.step()
            
            output_track.append(output)

            n = list(target.size())
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
                loss = F.mse_loss(output.view(-1, output.size(-1)), true_vals[:,i].float()) / normalize

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