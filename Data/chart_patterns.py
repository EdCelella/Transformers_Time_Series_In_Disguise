from .preprocessing import Batch, read_file, construct_sequence, build_tensors
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelmin, argrelmax
import code

# def main():
# 	df = read_file("S&P_500.csv")
# 	y = df["Close"].iloc[-200:-1].to_numpy()
# 	plot_price(y)
# 	vals, indx = get_peaks(y)
# 	t1 = [i[0] for i in indx]
# 	plot_peaks(vals, t1)
# 	plt.show()
# 	code.interact(local=dict(globals(), **locals()))

def main():
	data_set = get_cp_dataset()
	code.interact(local=dict(globals(), **locals()))

patterns = {
	"hs": lambda x, y: head_shoulders(x, y),
	"ihs": lambda x, y: inv_head_shoulders(x, y),
	"bt": lambda x, y: broad_top(x, y),
	"bb": lambda x, y: broad_bottom(x, y),
	"tt": lambda x, y: triangle_top(x, y),
	"tb": lambda x, y: triangle_bottom(x, y),
	"rt": lambda x, y: rectangle_top(x, y),
	"rb": lambda x, y: rectangle_bottom(x, y)
}

def get_cp_dataset(df, seq_size=10, pred_window=100, batch_size=64, step=1, pat_seq=90, test_size=0.1):

	global patterns
	cols = list(patterns.keys())

	df = construct_sequence(df, pat_seq, 0, step).astype(float)
	df = df.dropna(0).reset_index(0, drop=True)

	pat_df = df.apply(lambda row: detect(row.values), axis=1)
	c = pd.DataFrame(df.iloc[:, -1])
	
	features = pat_df.shape[1]
	inp = construct_sequence(pat_df, seq_size, 0, step).astype(float)
	out = construct_sequence(pat_df, seq_size, pred_window, step).astype(float)
	c = construct_sequence(c, seq_size, pred_window, step).astype(float)

	t1 = pd.concat([inp,out,c], axis=1).dropna(0)
	t1 = t1.sample(frac=1).reset_index(0, drop=True)

	inp = t1.values[:, :(seq_size*features)]
	out = t1.values[:, seq_size*features:-seq_size]
	c = t1.values[:, -seq_size:]

	assert(len(inp) == len(out))
	n_samples = len(inp)

	inp = build_tensors(inp, n_samples, seq_size, features, batch_size)
	out = build_tensors(out, n_samples, seq_size, features, batch_size)
	true = build_tensors(prod_class(c), n_samples, seq_size, 3, batch_size)

	data_set = []
	for i in range(len(inp)):
		data_set.append(Batch(inp[i], out[i], true[i]))

	return data_set

def prod_class(x):
	t1 = []
	for i in x:
		row = [0,1,0]
		prev = i[0]
		bought = False
		for j in range(0, len(i)-1):
			if i[j] > prev and bought == False:
				row.extend([1,0,0])
				bought = True
			elif i[j] < prev and bought == True:
				row.extend([0,0,1])
				bought = False
			else:
				row.extend([0,1,0])
			prev = i[j]
		t1.append(row)
	return np.array(t1)
	

def detect(row):

	global patterns
	n_pat = len(patterns)
	peaks_req = 5

	vals, indx = get_peaks(row)

	if len(vals) < peaks_req: return [False] * n_pat 

	detected = []
	for f in patterns.values():
		d = False
		for i in range(0, len(vals)-peaks_req):
			E, is_peak = vals[i:i+peaks_req], [j[1] for j in indx[i:i+5]]
			d = f(E, is_peak)
			if d: break
		detected.append(d)

	return pd.Series(detected)

def plot_price(data):
	x = np.arange(1, len(data)+1)
	plt.plot(x,data)

def plot_peaks(data, x):
	plt.scatter(x,data, c='red')

def get_peaks(data, window=3):
	peak_ind = argrelmax(data, order=window)[0]
	trough_ind = argrelmin(data, order=window)[0]
	indx = [[i, True] for i in peak_ind] + [[i, False] for i in trough_ind]
	indx.sort()
	return [data[i[0]] for i in indx], indx

def head_shoulders(E, is_peak):

	if len(E) < 5: return False

	if is_peak[0] == False: return False
	if E[2] <= E[0]: return False
	if E[2] <= E[4]: return False
	if within_mean_range(E[0], E[4]) is not True: return False
	if within_mean_range(E[1], E[3]) is not True: return False

	return True

def inv_head_shoulders(E, is_peak):

	if len(E) < 5: return False

	if is_peak[0] == True: return False
	if E[2] >= E[0]: return False
	if E[2] >= E[4]: return False
	if within_mean_range(E[0], E[4]) is not True: return False
	if within_mean_range(E[1], E[3]) is not True: return False

	return True

def broad_top(E, is_peak):

	if len(E) < 5: return False

	if is_peak[0] == False: return False
	if E[0] >= E[2] or E[2] >= E[4]: return False
	if E[1] <= E[3]: return False

	return True

def broad_bottom(E, is_peak):

	if len(E) < 5: return False

	if is_peak[0] == True: return False
	if E[0] <= E[2] or E[2] <= E[4]: return False
	if E[1] >= E[3]: return False

	return True

def triangle_top(E, is_peak):

	if len(E) < 5: return False

	if is_peak[0] == False: return False
	if E[0] >= E[2] or E[2] >= E[4]: return False
	if E[1] >= E[3]: return False

	return True

def triangle_bottom(E, is_peak):

	if len(E) < 5: return False

	if is_peak[0] == True: return False
	if E[0] <= E[2] or E[2] <= E[4]: return False
	if E[1] <= E[3]: return False

	return True

def rectangle_top(E, is_peak):
	
	if len(E) < 5: return False

	if is_peak[0] == False: return False

	peaks, troughs = [], []
	for i, e in enumerate(E):
		if is_peak[i] == True: peaks.append(e)
		else: troughs.append(e)

	if len(peaks) == 0 or len(troughs) == 0: return False
	if series_within_mean_range(peaks) == False: return False
	if series_within_mean_range(troughs) == False: return False
	if min(peaks) < max(troughs): return False
	return True

def rectangle_bottom(E, is_peak):
	
	if len(E) < 5: return False

	if is_peak[0] == True: return False

	peaks, troughs = [], []
	for i, e in enumerate(E):
		if is_peak[i] == True: peaks.append(e)
		else: troughs.append(e)

	if series_within_mean_range(peaks) == False: return False
	if series_within_mean_range(troughs) == False: return False
	if min(peaks) < max(troughs): return False
	return True


def within_mean_range(E1, E2, percent=0.015):
	m = (E1 + E2) / 2
	step = m * percent
	if E1 < m - step or E1 > m + step: return False
	if E2 < m - step or E2 > m + step: return False
	return True

def series_within_mean_range(E, percent=0.75):
	m = sum(E)/len(E)
	step = m * percent
	for e in E:
		if e < m-step or e > m+step: return False
	return True

if __name__ == "__main__":
	main()
