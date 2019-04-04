from __future__ import division, print_function
import numpy as np
import scipy
import scipy.io.wavfile as siow
import os
import time

DATA_NAME = 'WAV_11025'
DATA_ROOT_PATH = '/media/bighdd7/irene/HLSTM/data/%s/'%DATA_NAME
RAW_DATA_PATH = DATA_ROOT_PATH + 'raw/train/'
STORE_PATH = DATA_ROOT_PATH + 'spect/'

FREQ_SIZES = np.array([25, 50, 100, 200])
MAX_FREQ_SIZE = np.amax(FREQ_SIZES)

def read_data(filename):
	(rate, data) = siow.read(filename)
	if (len(data.shape) != 1): data = data[:, 0]
	return rate, data

# Convert raw data to spectrogram
# raw_data: a 1D array
# freq_size: 
def raw_to_spect(raw_data, freq_size):
	n_points = len(raw_data)
	assert(n_points % freq_size == 0)

	length = n_points // freq_size
	spectrogram = []

	for i in range(length):
		start = freq_size * i
		end = start + freq_size

		frame = raw_data[start:end]
		frame_fft = np.fft.fft(frame, axis=0)

		# Split real and imaginary parts
		real = frame_fft.real
		imag = frame_fft.imag
		frame_fft = np.concatenate((real, imag))

		spectrogram.append(frame_fft)

	spectrogram = np.array(spectrogram)

	return spectrogram

def get_weights(spect):
	# spect: [time_steps, input_dim]
	max_vals = np.amax(spect, axis=0).astype(float)
	min_vals = np.amin(spect, axis=0).astype(float)
	# max_vals/min_vals: [input_dim]
	# print(max_vals, min_vals)
	weights = 1/(max_vals - min_vals)**2
	np.place(weights, np.isinf(weights), 0.)
	norm_factor = np.sum(weights)
	if (norm_factor == 0.):
		# weights = np.zeros(weights.shape)
		input_dim = weights.shape[0]
		weights = np.repeat(1./input_dim, input_dim)
	else:
		weights = weights / norm_factor
	# weights: [input_dim]
	return weights

def get_file_name(file):
	index = file.find('.wav')
	file_name = file[:index]
	return file_name

def get_freq_path(freq_size):
	return STORE_PATH + '/frame_%d/'%freq_size

def store_video_data(data_path, file):
	file_name = get_file_name(file)
	file_path = data_path + file
	rate, raw_data = read_data(file_path) 
	assert(rate == 16000)
	print(file_name, raw_data.shape, end=' ')

	# Prune data
	length = len(raw_data)
	pruned_length = length - (length % MAX_FREQ_SIZE)
	raw_data = raw_data[:pruned_length]
	for freq_size in FREQ_SIZES:
		freq_path = get_freq_path(freq_size)
		store_file_path = freq_path + file_name + '.npy'
		if (os.path.exists(store_file_path)): continue
		spect = raw_to_spect(raw_data, freq_size)
		np.save(store_file_path, spect)

def store_data(data_path):
	if (not os.path.exists(STORE_PATH)): os.makedirs(STORE_PATH)
	for freq_size in FREQ_SIZES:
		freq_path = get_freq_path(freq_size)
		if (not os.path.exists(freq_path)): os.makedirs(freq_path)

	files = os.listdir(data_path)
	for file in files:
		if (file.endswith('.wav')): 
			start_time = time.time()
			store_video_data(data_path, file)
			elapsed_time = time.time() - start_time
			print(elapsed_time)

if __name__ == "__main__":

	# files = os.listdir(RAW_DATA_PATH)
	# print('Number of files: ', len(files))
	# i = 0
	# for file in files:
	# 	if (not file.endswith('.wav')): continue
	# 	if (file == 'vyB00TXsimI.wav'):
	# 		print('index', i)
	# 	i += 1
	# print(i)

	store_data(RAW_DATA_PATH)


################ Testing block ##################

# start_time = time.time()
# length = int(16e7)
# freq_size = 100
# data = np.random.rand(length)
# time_steps = length // freq_size
# data = data.reshape((time_steps, freq_size))
# elapsed_time = time.time() - start_time
# print(elapsed_time)


