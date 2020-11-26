import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import code

# class Regression_Output(nn.Module):

# 	def __init__(self, d_model, out_seq=1, hidden_layer=128, dropout=0.1):
# 		super(Regression_Output, self).__init__()
# 		self.layer_in = nn.Linear(d_model, hidden_layer)
# 		self.layer_hidden = nn.Linear(hidden_layer, out_seq)
# 		self.layer_out = nn.Linear(d_model, out_seq)
# 		self.drop = nn.Dropout(dropout)

# 	def forward(self, x):
# 		x = self.drop(self.layer_in(F.relu(x)))
# 		x = F.relu(self.layer_hidden(x)).squeeze(-1)
# 		# code.interact(local=dict(globals(), **locals()))
# 		return F.relu(self.layer_out(x))

class Regression_Output(nn.Module):

	def __init__(self, d_model, hidden_layer=128, dropout=0.1):
		super(Regression_Output, self).__init__()
		self.layer_in = nn.Linear(d_model, hidden_layer)
		self.layer_hidden = nn.Linear(hidden_layer,hidden_layer)
		self.layer_out = nn.Linear(hidden_layer, d_model)
		self.drop = nn.Dropout(dropout)

	def forward(self, x):
		x = self.drop(self.layer_in(F.relu(x)))
		x = self.drop(self.layer_hidden(F.relu(x)))
		return F.relu(self.layer_out(x))
		# return self.layer_out(F.relu(x))

class Classification_Output(nn.Module):

    def __init__(self, d_model, dec_class=3):
        super(Classification_Output, self).__init__()
        self.layer = nn.Linear(d_model, dec_class)

    def forward(self, x):
        return F.softmax(self.layer(x), dim=-1)

# class Regression_Output(nn.Module):

# 	def __init__(self, d_model, out_seq=1, hidden_layer=128, dropout=0.1):
# 		super(Regression_Output, self).__init__()
# 		self.layer_in = nn.Linear(d_model, hidden_layer)
# 		self.layer_hidden1 = nn.Linear(hidden_layer, hidden_layer)
# 		self.layer_hidden2 = nn.Linear(hidden_layer, hidden_layer)
# 		self.layer_out = nn.Linear(hidden_layer, out_seq)
# 		self.drop1 = nn.Dropout(dropout)
# 		self.drop2 = nn.Dropout(dropout)
# 		self.drop3 = nn.Dropout(dropout)

# 	def forward(self, x):
# 		x = self.drop1(self.layer_in(x))
# 		x = self.drop2(self.layer_hidden1(x))
# 		x = self.drop3(self.layer_hidden2(x))
# 		return self.layer_out(x)