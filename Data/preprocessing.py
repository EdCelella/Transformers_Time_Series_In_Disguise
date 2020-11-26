import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as ag
from random import shuffle
import pickle
import code
# code.interact(local=dict(globals(), **locals()))

MA_Functions = {
	"sma": lambda x: sum(x)/len(x), # Simple Moving Average
	"wma": lambda x: sum([x[i] * (i+1) for i in range(len(x))]) / sum(range(len(x)+1)), # Weighted Moving Average
	"ewma": lambda x, a=0.5: sum([x[i]*((1-a)**(len(x)-i)) for i in range(len(x))]) / sum([(1-a)**i for i in range(len(x))]) # Exponentially Weighted Moving Average
}

class Batch:
	def __init__(self, enc_inp, dec_inp, true_vals, pad_tok=0):
		self.enc_inp = enc_inp
		self.dec_inp = dec_inp
		self.pad_tok = pad_tok
		self.enc_mask = self.build_enc_mask()
		self.dec_mask = self.build_dec_mask()
		self.true_vals = true_vals

	def build_enc_mask(self): 
		return (self.enc_inp[:,:,0] != self.pad_tok).unsqueeze(-2)

	def build_dec_mask(self):
		seq_size = self.dec_inp.size()[-2]
		mask = (self.dec_inp[:,:,0] != self.pad_tok).unsqueeze(-2)
		t1 = np.triu(np.ones((1, seq_size, seq_size)), k=1).astype('uint8')
		t1 = torch.from_numpy(t1) == 0
		return ag.Variable(mask & t1)

	def move_to_device(self, device):
		self.enc_inp = self.enc_inp.to(device=device)
		self.dec_inp = self.dec_inp.to(device=device)
		self.enc_mask = self.enc_mask.to(device=device)
		self.dec_mask = self.dec_mask.to(device=device)
		self.true_vals = self.true_vals.to(device=device)

def read_file(filename):
	df = pd.read_csv(filename, delimiter=',', encoding="utf-8", skipinitialspace=True)
	return df.iloc[::-1].reset_index(drop=True)

def filter_date(df, date='31-12-2009'):
	df['Date'] = pd.to_datetime(df['Date'])
	return df.loc[df['Date'] > date].reset_index(0, drop=True)

def calc_moving_average(df, N=20, f="sma", col='Close'):

	if N != None:
		new_col = f + "_" + str(N)
		df[new_col] = df[col].rolling(N).apply(MA_Functions[f], raw=True)
	else: 
		df[f] = df[col].expanding().apply(MA_Functions[f], raw=True)
	return df

def save_data(data_set, filename):
	with open(filename, "wb") as f:
		pickle.dump(data_set, f)

def load_data(filename):
	with open(filename, "rb") as f:
		data_set = pickle.load(f)
	return data_set

###############################################################################################

def build_dataset(df, seq_size=5, pred_window=5, batch_size=64, step=1):

	features = df.shape[1]

	inp = construct_sequence(df, seq_size, 0, step).astype(float)
	out = construct_sequence(df, seq_size, pred_window, step).astype(float)
	
	t1 = pd.concat([inp,out], axis=1).dropna(0)
	t1 = t1.apply(lambda row: normalize(row.values, seq_size, features), axis=1)
	t1 = t1.sample(frac=1).reset_index(0, drop=True)

	inp = t1.values[:, :seq_size*features]
	out = t1.values[:, seq_size*features:]

	assert(len(inp) == len(out))
	n_samples = len(inp)

	inp = build_tensors(inp, n_samples, seq_size, features, batch_size)
	out = build_tensors(out, n_samples, seq_size, features, batch_size)

	data_set = []
	for i in range(len(inp)):
		data_set.append(Batch(inp[i], out[i], out[i]))

	return data_set

def construct_sequence(x, seq_size, pred_window, step=1):
	t1 = x[:].iloc[pred_window:].reset_index(0, drop=True)
	for i in range(pred_window+1, pred_window+seq_size):
		index = i * step
		t2 = x[:].iloc[index:].reset_index(0, drop=True)
		t1 = pd.concat([t1,t2],axis=1)
	return t1

def normalize(row, seq_size, features):
	inp = row[:seq_size*features]
	minimum = float(min(inp))
	r = float((max(inp)-min(inp)))
	for i in range(len(row)):
		row[i] = (row[i]-minimum)/r
	return pd.Series(row)

def build_tensors(x, n_samples, seq_size, features, batch_size):
	x = x.reshape(n_samples, seq_size, features)
	x = split_batches(x, batch_size)
	data_set = []
	for i in x:
		data_set.append(torch.from_numpy(i).float())
	return data_set

def prod_class(data_set, close_dim=3):
	for b in data_set:
		x = b.true_vals
		c = torch.zeros([x.size(0), x.size(1), 3])
		prev = 1
		bought = False
		for i, seq in enumerate(x):
			for j, p in enumerate(seq):
				if p[close_dim].item() > prev and bought == False:
					c[i][j][0] = 1
					bought = True
				elif p[close_dim].item() < prev and bought == True:
					c[i][j][2] = 1
					bought = False
				else:
					c[i][j][1] = 1
				prev = p[close_dim]
		b.true_vals = c
	return data_set
	

###############################################################################################


def get_raw_dataset(filename="S&P_500.csv", seq_size=5, pred_window=5, batch_size=64, step=1):
	df = read_file(filename)
	train, test = split_data(df)
	train = build_dataset(train[['Open', 'High', 'Low', 'Close']], seq_size=seq_size, pred_window=pred_window, batch_size=batch_size, step=step)
	test = build_dataset(test[['Open', 'High', 'Low', 'Close']], seq_size=seq_size, pred_window=pred_window, batch_size=batch_size, step=step)
	return train, test

def get_ma_dataset(filename="S&P_500.csv", ma="wma", N_low=20, N_high=120, step=20, seq_size=5, pred_window=5, batch_size=64, step_seq=1):
	df = read_file(filename)
	cols = ['Close']
	for i in range(N_low, N_high, step):
		df = calc_moving_average(df, N=i, f=ma)
		cols.append(ma+'_'+str(i))
	df = df.dropna(0).reset_index(0, drop=True)
	train, test = split_data(df)
	train = build_dataset(train[cols], seq_size=seq_size, pred_window=pred_window, batch_size=batch_size, step=step_seq)
	test = build_dataset(test[cols], seq_size=seq_size, pred_window=pred_window, batch_size=batch_size, step=step_seq)
	return train, test

def split_batches(x, bs):
	if len(x) <= bs: return [x]
	return [x[:bs]] + split_batches(x[bs:],bs)

def split_data(df, test_prop=0.1):
	n_samples = df.shape[0]
	split = n_samples - int(n_samples * test_prop)
	return df.iloc[:split], df.iloc[split:]

def test():
	d1_train, d1_test = get_raw_dataset()
	code.interact(local=dict(globals(), **locals()))



###############################################################################################
if __name__ == "__main__":
	test()

# def get_raw_dataset(filename="Data/S&P_500.csv"):
# 	df = read_file(filename)
# 	return batch_data(df[len(df)-2665:], input_cols=["Open", "High", "Low", "Close"])

# def batch_data(df, input_cols, output_cols=["Close"], batch_size=64, seq_size=5, prediction=(6, 4)):

# 	num_points = df.shape[0]

# 	assert(num_points % seq_size == 0)

# 	inputs = []
# 	step = batch_size * seq_size
# 	for i in range(0, num_points-step):
# 		temp = df.iloc[i:i+step]
# 		inputs.append(construct_tensor(temp, input_cols, seq_size))

# 	outputs = get_output(df, prediction, num_points, seq_size, batch_size)
	
# 	data_set = []
# 	for i in range(len(outputs)):
# 		data_set.append(Batch(inputs[i], inputs[i+seq_size+1], outputs[i]))

# 	return data_set


# def construct_tensor(df, input_cols, seq_size):

# 	tensors = []
# 	for i in input_cols:
# 		tensors.append(torch.tensor(df[i].values))
# 	data_set = torch.stack(tensors, dim=1).view(-1,1,len(input_cols))
# 	return F.normalize(data_set.view(-1,seq_size,len(input_cols)), dim=1)

# def get_output(df, prediction, num_points, seq_size, batch_size, output_cols=["Close"]):

# 	pred_indx = seq_size + prediction[0]
# 	step = seq_size * batch_size

# 	outputs = []
# 	for i in range(pred_indx, num_points-step):
# 		temp = []
# 		for j in range(i, i+step, seq_size):
# 			temp.append(df[output_cols].iloc[j:j+prediction[1],].values.squeeze(-1))
# 		outputs.append(torch.tensor(temp))
# 	return outputs

# def split_data(data_set, test_prop=0.1):
# 	test_size = int(len(data_set) * test_prop)
# 	training_set = data_set[:len(data_set)-test_size]
# 	test_set = data_set[len(data_set)-test_size: len(data_set)]
# 	shuffle(training_set)
# 	return training_set, test_set
















