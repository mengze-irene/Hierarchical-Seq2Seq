from __future__ import division, print_function
import numpy as np 
import os, time, math, csv
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

np.warnings.filterwarnings('ignore')

SPECT_PATH = "/media/bighdd7/irene/HLSTM/data/WAV_16000/spect/"
test_video_id = '6KvG5VbLalY'

# data: [n_instances, time_steps, input_dim]
# Normalize per instance
def normalize(data):
	min_vals = np.amin(np.amin(data, axis=-1), axis=-1)
	min_vals = np.expand_dims(np.expand_dims(min_vals, -1), -1)
	max_vals = np.amax(np.amax(data, axis=-1), axis=-1).astype(float)
	max_vals = np.expand_dims(np.expand_dims(max_vals, -1), -1).astype(float)
	denom = max_vals - min_vals
	data = (data - min_vals) / (max_vals - min_vals) # [0, 1]
	# Would results in inf. or nan.

	# Get rid of inf and nan. 
	np.place(data, np.isinf(data), [0.5])
	np.place(data, np.isnan(data), [0.5])

	data = (data - 0.5)*2*0.9 # [-0.9, 0.9]

	return data, min_vals, max_vals


def denormalize(outputs, min_vals, max_vals):
	outputs /= 0.9 # [-1.0, 1.0]
	outputs = outputs *0.5 + 0.5 # [0., 1.]
	outputs = outputs * (max_vals - min_vals) + min_vals # [min_vals, max_vals]

	return outputs

# data: [-1, input_dim]
def preprocess(data, time_steps, return_min_max=False):
	length, input_dim = data.shape[0], data.shape[-1]
	pruned_len = length - length % time_steps
	data = data[:pruned_len]
	data = data.reshape(-1, time_steps, input_dim)

	(data, min_vals, max_vals) = normalize(data)
	# min/max: [n_instances]

	data = torch.tensor(data, dtype=torch.float)

	if (return_min_max): 
		return (data, min_vals, max_vals)
	else:
		return data

# DEPRECATED
def split(X, valid_ratio=0.1):
	n_videos = len(X)
	indices = np.arange(n_videos)

	valid_size = int(round(n_videos * valid_ratio))
	valid_indices = np.random.choice(indices, valid_size, replace=False)

	mask = np.array([(i in valid_indices) for i in indices])

	valid_data = X[mask]
	train_data = X[~mask]

	return train_data, valid_data

# DEPRECATED
# Return (train_data, valid_data, (test_data, min_vals, max_vals)):
# train_data, valid_data and test_data are 3D tensors of shape
# (n_instances, time_steps, input_dim) and are normalized per instance.
# min_vals is a 1D array of min vals for each instance in test_data.
# max_vals is similar.
def load_data(freq_size, time_steps, test_only):
	freq_path = SPECT_PATH + 'frame_%d/'%freq_size
	print(freq_path)

	X = []
	files = os.listdir(freq_path)

	# Load test data
	test_file = files[0]
	print('Test file', test_file)
	test_data = np.load(freq_path + test_file)
	test_set = preprocess(test_data, time_steps)
	if (test_only): return None, None, test_set

	start_time = time.time()
	# Load train and valid data
	for i in range(1, len(files)):
		file = files[i]
		if (file.endswith('.npy')):
			data = np.load(freq_path + file)
			print(file, data.shape)
			if (np.isnan(data).any()): print('has NAN!!!')
			X.append(data)
			if (i % 200 == 0):
				elapsed_time = time.time() - start_time
				print('%d files loaded after %f.'%(i, elapsed_time))

	# Split before concatenating to avoid speaker overlapping
	train_data, valid_data = split(np.array(X), valid_ratio=0.1)
	train_data, _, _ = preprocess(np.concatenate(train_data), time_steps)
	valid_data, _, _ = preprocess(np.concatenate(valid_data), time_steps)

	return train_data, valid_data, test_set

def load_test_data(input_dim, time_steps):
	freq_size = input_dim//2
	freq_path = SPECT_PATH + 'frame_%d/'%freq_size

	# Load test data
	test_file = '6KvG5VbLalY.npy'
	test_data = np.load(freq_path + test_file)
	test_set = preprocess(test_data, time_steps, return_min_max=True)

	return test_set

class SingleVideoDataset(Dataset):
	"""docstring for SingleVideoDataset"""
	def __init__(self, root_dir, video_id, data_size, time_steps, input_dim):
		super(SingleVideoDataset, self).__init__()
		self.root_dir = root_dir
		self.video_id = video_id
		self.data_size = data_size
		self.time_steps = time_steps
		self.input_dim = input_dim

		total_time_steps = data_size // input_dim
		self.len = total_time_steps // time_steps

		self.data = None
		self.loaded = False

	def __len__(self):
		return self.len 

	def __getitem__(self, idx):
		if (not self.loaded):
			data = np.load(self.root_dir + self.video_id + '.npy')
			self.data = preprocess(data, self.time_steps)
			self.loaded = True
			# if (self.data.shape[0] == self.len): 
			#print(self.video_id, self.data.shape, self.len)

		# print(self.video_id, self.data.shape, idx, self.len)
		return self.data[idx]

csvfile = '/media/bighdd7/irene/HLSTM/data/WAV_16000/spect/video_length.csv'

def get_data_loader(csvfilename, input_dim, time_steps, batch_size, shuffle):
	freq_size = input_dim//2
	root_dir = SPECT_PATH + 'frame_%d/'%freq_size
	
	datasets = []
	with open(SPECT_PATH+csvfilename, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			video_id = row[0]
			data_size = int(row[1])
			video_dataset = SingleVideoDataset(root_dir, video_id, data_size,
				time_steps, input_dim)
			datasets.append(video_dataset)

	data_loader = DataLoader(
             ConcatDataset(datasets),
             batch_size=batch_size, shuffle=shuffle,
             num_workers=0, pin_memory=False)

	return data_loader

if __name__ == '__main__':
	input_dim = 50
	time_steps = 256
	batch_size = 256
	data_loader = get_data_loader('valid_length.csv', input_dim, 
		time_steps, batch_size, False)

	start_time = time.time()
	for i_batch, sample_batched in enumerate(data_loader):
		print(i_batch, sample_batched.shape)
		if (i_batch == 9): print(time.time() - start_time)

