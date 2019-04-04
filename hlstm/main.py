from __future__ import division, print_function
import numpy as np
import os, sys, time, math
from collections import defaultdict
import pickle
import h5py
import scipy
import scipy.io.wavfile as siow

import csv
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

# from models import encdec_model
from encdec_model import enc_dec
from loss import LapLoss
from process_data import get_weights
from settings import GS_FIELDNAMES, LOSS_NAMES, FINAL_CRTR, DEVICE, CSVFILENAME, ARCH_DIR
from data_loader import normalize, denormalize, load_test_data, get_data_loader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cmu_mosi_std_folds as std

train_fold = std.standard_train_fold
valid_fold = std.standard_valid_fold
test_fold = std.standard_test_fold

# Representation Learning
# python main.py --learn_mode

print("Device:", DEVICE)
################ Hyper parameter section ##########################
parser = argparse.ArgumentParser()
parser.add_argument("--input_dim", help="set input dimension", default=100, type=int)
parser.add_argument("--time_steps", help='set time steps', default=256, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 512, 512])
parser.add_argument('--n_splits', nargs='+', type=int, default=[32, 8, 1])
parser.add_argument("--learning_rate", '-lr', default=0.001, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("-r", "--reverse", action="store_true")
parser.add_argument("--dropout", default=0., type=float)
parser.add_argument("--loss_weights", nargs='+', type=int, default=[1, 0, 0, 0])
parser.add_argument("--no_training", action="store_true")
parser.add_argument("--test_mode", action="store_true")
parser.add_argument("--learn_mode", action="store_true")
parser.add_argument("-v", "--video_id")

parser.add_argument("--resume", action="store_true")

parser.add_argument("--config", type=int, default=42)
parser.add_argument("--weights_file", "-wf")

parser.add_argument("-p", "--pseudo", action="store_true")
parser.add_argument("--init_weights", "-iw")
parser.add_argument("--no_command", action='store_true')
parser.add_argument("--parent_dir", default=ARCH_DIR + 'test_space/')

# Deprecated arguments. Remain for compatibility reason.
parser.add_argument("--train_mode", action='store_true')

args = parser.parse_args()

INPUT_DIM = args.input_dim
TIME_STEPS = args.time_steps
BATCH_SIZE = args.batch_size
HIDDEN_DIMS = args.hidden_dims
N_SPLITS = args.n_splits
LR = args.learning_rate
N_LEVELS = len(HIDDEN_DIMS)
N_EPOCHS = args.epochs
REVERSE = args.reverse
SELF_NORM = False #args.self_norm
DROPOUT = args.dropout
LOSS_WEIGHTS = args.loss_weights
MSE_WEIGHTS = LOSS_WEIGHTS[0]
MAE_WEIGHTS = LOSS_WEIGHTS[1]
CORR_WEIGHTS = LOSS_WEIGHTS[2]
LAP_WEIGHTS = LOSS_WEIGHTS[3]

TOTAL_LOSS_WEIGHTS = sum(LOSS_WEIGHTS)

TRAIN_MODE = not args.no_training
TEST_MODE = args.test_mode
LEARN_MODE = args.learn_mode
VIDEO_ID =args.video_id
RESUME = args.resume

CONFIG_NUM = args.config
WEIGHTS_FILE = args.weights_file
PARENT_DIR = args.parent_dir
CONFIG_DIR = PARENT_DIR + 'config%d/'%CONFIG_NUM 
if (not os.path.exists(PARENT_DIR)): os.makedirs(PARENT_DIR)
if (not os.path.exists(CONFIG_DIR)): os.makedirs(CONFIG_DIR)
LAST_WEIGTHS_PATH = CONFIG_DIR+'last_weights.pt'
STATUS_PATH = CONFIG_DIR + 'status.p'
WEIGHTS_PATH = (CONFIG_DIR + WEIGHTS_FILE) if (WEIGHTS_FILE != None) else None
CSVFILEPATH = PARENT_DIR + CSVFILENAME

INIT_WEIGHTS = args.init_weights
STORE_COMMAND = not args.no_command

STORE_RESULTS = True # !!!!! Be careful!
PSEUDO = args.pseudo

DATA_PATH = "/media/bighdd7/irene/HLSTM/data/WAV_16000/spect"

# Store hyperparameters in config directory
command = 'python main.py ' + ' '.join(sys.argv[1:])
if (TRAIN_MODE and STORE_COMMAND):
	hp_filename = CONFIG_DIR + 'hyperparams.txt'
	file = open(hp_filename, 'w')
	file.write(command)
	file.close()

if (PSEUDO or (N_EPOCHS < 10) or not TRAIN_MODE):
	print('NOT READY FOR REAL GRID SEARCH!!!')

KNOB = 10

print("#####################################")
print("HYPERPARAMETERS: ")
print("Conifg num = %d"%CONFIG_NUM)
print("Learning rate=%f"%LR)
print("Input dim = %d"%INPUT_DIM)
print("Time steps = %d"%TIME_STEPS) 
print("Batch size = %d"%BATCH_SIZE)
print("Levels = %d"%N_LEVELS)
print("Hidden dims: ", HIDDEN_DIMS)
print("Splits: ", N_SPLITS)
print("Reverse: ", REVERSE)
print("Self-normalization: ", SELF_NORM)
print("Loss weights: ", LOSS_WEIGHTS)
print("Train mode " + ("on" if TRAIN_MODE else "off" ))
print("Test mode " + ("on" if TEST_MODE else "off"))
print("Representation Learning mode " + ("on" if LEARN_MODE else "off"))
print("#####################################")

params_dict = {'config_num' : CONFIG_NUM, 
				'input_dim': INPUT_DIM, 'time_steps': TIME_STEPS, 'n_levels': N_LEVELS,
                'hidden_dims': HIDDEN_DIMS, 'n_splits': N_SPLITS, 'lr': LR, 
                'reverse' : REVERSE, 'batch_size' : BATCH_SIZE, 'loss_weights': LOSS_WEIGHTS, 
                'dropout' : DROPOUT}

dict_keys = sorted(params_dict.keys())
gs_keys = sorted(GS_FIELDNAMES[:len(dict_keys)])
assert(gs_keys == dict_keys)

################ Hyper parameter section Ends ##########################
metric_names = []
for loss_name in LOSS_NAMES:
	metric_names.append('val_'+loss_name)
	metric_names.append('val_'+loss_name+'_e2e')
metrics_fieldnames = ['level','epoch','best_yet', 'training_loss'] + metric_names
metrics_filename = CONFIG_DIR + 'metrics.csv'

################ Utility functions ###################

def anynan(x):
	return torch.isnan(x).any().item()

def anynan_in_list(A):
	for x in A:
		if (anynan(x) != 0): return True
	return False

# Split torch tensor into two sub-tensors according to ratio
def split(iterator, ratio):
	size = iterator.size()[0]

	split_point = int(size * ratio)

	return iterator[:split_point], iterator[split_point:]

# Given numpy array x, reshape x by adding new_dim as
# x's 1st dimension. 
# Padding zeros when shape doesn't agree with desired shape.
def myReshape(x, new_dim):
	old_shape = x.shape
	length = old_shape[0]
	overflow = length % new_dim
	if (overflow != 0):
		gap_len = new_dim - overflow
		padd_shape = [gap_len] + list(old_shape[1:])
		padding = np.zeros(padd_shape)
		x = np.concatenate((x, padding))
	new_shape = [-1, new_dim]
	for i in range(1, len(old_shape)):
		dim = old_shape[i]
		new_shape.append(dim)

	x = np.reshape(x, new_shape)
	return x

def write_results(filepath, criterion_vals, best_epoch, best_epochs):
	marker = '.'
	with open(filepath, 'at') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=GS_FIELDNAMES)
		contents = dict()
		for fieldname in GS_FIELDNAMES:
			if (fieldname in criterion_vals): val = criterion_vals[fieldname]
			elif (fieldname == 'best_epoch'): val = best_epoch
			elif (fieldname == 'best_epochs'): val = best_epochs
			elif (fieldname in params_dict): val = params_dict[fieldname]
			else: raise Exception('Unknown fieldname', fieldname)
			contents[fieldname] = val
		writer.writerow(contents)

def write_epoch_result(info):
	with open(metrics_filename, 'at') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=metrics_fieldnames)
		writer.writerow(info)

################ Utility functions ###################

################ Data processing #####################


# Convert spectrogram to raw data
# spectrogram: 2D numpy array
# output: 1D numpy array
def spect_to_raw(spectrogram):
	shape = spectrogram.shape
	# assert(len(shape) == 2)
	time_steps = shape[0]
	frame_size = shape[-1]

	raw_data = []

	for i in range(time_steps):
		frame_fft = spectrogram[i]
		real = frame_fft[:(frame_size//2)]
		imag = frame_fft[(frame_size//2):]
		frame_fft = np.add(real, imag * 1j)

		frame = np.fft.ifft(frame_fft)

		raw_data.append(frame)

	raw_data = np.array(raw_data)
	raw_data = raw_data.real
	raw_data = raw_data.astype(np.int16)
	raw_data = raw_data.flatten()
	return raw_data

def post_process(outputs, min_vals, max_vals):
	if (not SELF_NORM):
		outputs = denormalize(outputs, min_vals, max_vals)
	# print(outputs)
	# outputs: [-1 TIME_STEPS, INPUT_DIM]
	spectrogram = np.reshape(outputs, (-1, INPUT_DIM))
	raw_data = spect_to_raw(spectrogram)
	# raw_data = raw_data.astype(np.int16)

	return raw_data

################# Data Processing #################

################# Loss Functions ##################

# x/y: [batch_size, time_steps, input_dim]
def corr_loss(x, y):
	x_mean = torch.mean(x, -1, True)
	y_mean = torch.mean(y, -1, True)
	# x_mean/y_mean: [batch_size, input_dim, 1]
	x_diff = x - x_mean
	y_diff = y - y_mean
	# x_diff/y_diff: [batch_size, input_dim, time_steps]
	numerator = torch.mean(x_diff * y_diff, -1)
	# numerator: [batch_size, input_dim]

	varx_list = x_diff ** 2
	vary_list = y_diff ** 2
	# varx_list/vary_list: [batch_size, input_dim, time_steps]

	varx = torch.mean(varx_list, -1)
	vary = torch.mean(vary_list, -1)
	# varx/vary: [batch_size, input_dim]
	sigx = torch.sqrt(varx)
	sigy = torch.sqrt(vary)
	denom = sigx * sigy
	# denom: [batch_size, input_dim]

	corr = numerator / denom
	corr = torch.mean(corr)

	return (1.-corr).to(DEVICE)

# x/y: [batch_size, time_steps, input_dim]
def twofold_corr_loss(x, y):
	# Add a small noise
	# Create noise of the same shape as x, y, with a mean of 1 and 
	# a standard deviation of 1e-5
	noise_stddev = 1e-5
	x_noise = (torch.randn(x.size()) * noise_stddev).to(DEVICE)
	y_noise = (torch.randn(x.size()) * noise_stddev).to(DEVICE)
	x = x + x_noise
	y = y + y_noise
	
	time_corr_loss = corr_loss(x, y)

	x, y = x.transpose(-2, -1), y.transpose(-2, -1)
	# x/y: [batch_size, input_dim, time_steps]
	freq_corr_loss = corr_loss(x, y)

	return (time_corr_loss + freq_corr_loss)/2

# Return a matrix of squared error between each element.
def mse_loss(x, y):
	# x/y: [batch_size, time_steps, input_dim]
	mse_fn = nn.MSELoss(reduce=False).to(DEVICE)
	mse_matrix = mse_fn(x, y) 
	# mse_matrix: [batch_size, time_steps, input_dim]

	return mse_matrix

# Return a matrix of absolute error between each element.
def mae_loss(x, y):
	mae_fn = nn.L1Loss(reduce=False).to(DEVICE)
	mae_matrix = mae_fn(x, y)

	return mae_matrix

max_levels = 3
(h_sigma, w_sigma) = (0.5, 0.8)
height = 3
width = 5
lap_loss = LapLoss(max_levels, (height, width), (h_sigma, w_sigma)).to(DEVICE)

# target and results are both 3D tensor of shape [batch_size, steps, dim].
# Return loss, which is a 3D tensor of the same shape as target and results.
def single_loss(target, results, criterion):
	#target.requires_grad_(False)
	target = torch.tensor(target.data, requires_grad=False)
	loss = criterion(results, target)
	# loss: [batch_size, time_steps, input_dim]
	return loss 

# enc_reps/dec_reps: [_, batch_size, _, _]
def level_loss(enc_reps, dec_reps, criterion, end2end):
	loss = 0
	# print("level loss:")
	levels = len(dec_reps)
	up_limit = 1 if end2end else levels
	counter = 0
	for i in range(up_limit):
		sub_loss_tensor = single_loss(enc_reps[i], dec_reps[i], criterion)
		if (sub_loss_tensor.nelement() == 0):
			continue

		sub_loss = torch.mean(sub_loss_tensor)
		loss += sub_loss
		counter += 1

	return loss/counter
		
	# if (counter == 0):
	# 	return torch.tensor([])
	# else:
	# 	return loss/counter

def full_loss(enc_reps, dec_reps_list, criterion, end2end):
	loss = 0
	length = len(dec_reps_list)
	counter = 0
	for i in range(length):
		dec_reps = dec_reps_list[i]
		loss_i = level_loss(enc_reps, dec_reps, criterion, end2end)
		if (loss_i.nelement() == 0): continue
		else:
			loss += loss_i
			counter += 1

	return loss/counter
	# if (counter == 0):
	# 	return torch.tensor([])
	# else:
	# 	return loss/counter

def all_losses(enc_reps, dec_reps, greedy, end2end):
	loss_fn = level_loss if greedy else full_loss
	mse = loss_fn(enc_reps, dec_reps, mse_loss, end2end)
	mae = loss_fn(enc_reps, dec_reps, mae_loss, end2end)
	corr = loss_fn(enc_reps, dec_reps, twofold_corr_loss, end2end)
	lap_pyrm = loss_fn(enc_reps, dec_reps, lap_loss, True)
	
	# Computed weighted average of these losses

	hybrid = (MSE_WEIGHTS * mse + MAE_WEIGHTS * mae 
		+ CORR_WEIGHTS * corr + LAP_WEIGHTS * lap_pyrm)
	hybrid /= TOTAL_LOSS_WEIGHTS

	# hybrid = corr

	loss_dict = {'mse': mse, 'mae': mae, 'corr': corr, 'lap': lap_pyrm, 'hybrid': hybrid}
	# return (mse * 10 + mae + corr)
	return loss_dict

def cake_losses(enc_reps, dec_reps, end2end):
	mse = full_loss(enc_reps, dec_reps, mse_loss, end2end)
	mae = full_loss(enc_reps, dec_reps, mae_loss, end2end)
	corr = full_loss(enc_reps, dec_reps, twofold_corr_loss, end2end)
	lap_pyrm = full_loss(enc_reps, dec_reps, lap_loss, True)
	
	# Computed weighted average of these losses

	hybrid = (MSE_WEIGHTS * mse + MAE_WEIGHTS * mae 
		+ CORR_WEIGHTS * corr + LAP_WEIGHTS * lap_pyrm)
	hybrid /= TOTAL_LOSS_WEIGHTS

	# hybrid = corr

	loss_dict = {'mse': mse, 'mae': mae, 'corr': corr, 'lap': lap_pyrm, 'hybrid': hybrid}
	# return (mse * 10 + mae + corr)
	return loss_dict

################# Loss Functions ##################

# Train on one epoch
# If not greedy, train levelth layer only
def train_epoch(model, train_loader, level, optimizer):
	start_time = time.time()

	epoch_loss = 0
	n_iters = 0
	teacher_forcing = True

	model.train()

	start_time = time.time()

	for i_batch, sample_batched in enumerate(train_loader):
		n_iters += 1
		# if (n_iters >= 3): break
		sample_batched = sample_batched.to(DEVICE)
		optimizer.zero_grad()

		enc_reps, dec_reps_list = model(sample_batched, teacher_forcing)

		end2end = False
		# Only compute loss on full encoding-decoding
		loss_dict = all_losses(enc_reps, dec_reps_list[-1], True, False)
		# loss_dict = cake_losses(enc_reps, dec_reps, end2end) 
		loss = loss_dict[FINAL_CRTR]

		# loss: singleton tensor
		loss.backward()
		optimizer.step()
		batch_loss = loss.item()
		epoch_loss += batch_loss

		if ((i_batch+1)%1==0):
			print("Batch %d: loss=%f time=%f"%(i_batch, batch_loss, time.time()-start_time))
			start_time = time.time()

	time_elapsed = time.time() - start_time

	return epoch_loss / n_iters, time_elapsed

def evaluate_epoch(model, valid_loader, level):
	start_time = time.time()

	model.eval()

	teacher_forcing = False

	criterion_vals = dict()
	for loss_name in LOSS_NAMES:
		criterion_vals[loss_name] = [0, 0]

	with torch.no_grad():

		n_iters = 0
		for i_batch, sample_batched in enumerate(valid_loader):
			n_iters += 1
			sample_batched = sample_batched.to(DEVICE)

			enc_reps, dec_reps_list = model(sample_batched, teacher_forcing)

			loss_dict = cake_losses(enc_reps, dec_reps_list, False)
			loss_end2end_dict = cake_losses(enc_reps, dec_reps_list, True)

			# loss_dict = all_losses(enc_reps, dec_reps[-1], True, True)
			# loss_end2end_dict = all_losses(enc_reps, dec_reps[-1], True, True)
			for key in criterion_vals:
				val_list = criterion_vals[key]
				val_list[0] += loss_dict[key].item()
				val_list[1] += loss_end2end_dict[key].item()

	for key in criterion_vals:
		val_list = criterion_vals[key]
		for i in range(len(val_list)):
			val_list[i] /= n_iters

	time_elapsed = time.time() - start_time
	print("Evaluation takes %.2fs"%time_elapsed)

	return criterion_vals, criterion_vals[FINAL_CRTR][1], time_elapsed


def reconstruct(data, min_vals, max_vals, filename):
	# data_batches: [N_BATCHS, BATCH_SIZE, TIME_STEPS, INPUT_DIM] numpy array
	raw_data = post_process(data, min_vals, max_vals)

	results_directory = CONFIG_DIR + "audio_results/"
	if (not os.path.exists(results_directory)):
		os.makedirs(results_directory)

	store_path = results_directory + filename
	freq = 16000
	siow.write(store_path, freq, raw_data)


def test(model, test_set, filename):
	#print("Testing")

	start_time = time.time()

	(test_data, min_vals, max_vals) = test_set
	test_data = test_data.to(DEVICE)

	model.eval()

	enc_reps, dec_reps = model.part_forward(test_data, False, N_LEVELS)
	output = dec_reps[0]
	# output: [-1, time steps, input dim]
	output = output.detach().cpu().numpy()

	reconstruct(output, min_vals, max_vals, filename)

######## Test decoder model ############
# HIDDEN_DIM = 174
# N_SPLIT = 4
# OUTPUT_DIM = 50
# MAX_VAL = 1000
# dec_model = decoderLayer(N_SPLIT, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE)
# x = torch.randint(0, MAX_VAL, (BATCH_SIZE, TIME_STEPS, INPUT_DIM), dtype=torch.float)
# h0 = torch.zeros((1, BATCH_SIZE*N_SPLIT, HIDDEN_DIM), dtype=torch.float)
# c0 = torch.zeros((1, BATCH_SIZE*N_SPLIT, HIDDEN_DIM), dtype=torch.float)

# out = dec_model(x, h0, c0, False, TIME_STEPS/N_SPLIT)
# print(out.size())

# print("Done testing decoder model")
# assert(False)
######## Test decode model ############

########### Test full model ################


# model = enc_dec(INPUT_DIM, BATCH_SIZE, N_LEVELS, HIDDEN_DIMS, N_SPLITS)
# # data = torch.randint(0, MAX_VAL, (BATCH_SIZE, TIME_STEPS, INPUT_DIM), dtype=torch.float)
# data = torch.rand(BATCH_SIZE, TIME_STEPS, INPUT_DIM)

# print("Inference")
# print("encoded representations")

# model.encode(data, N_LEVELS)
# enc_reps, dec_reps = model.part_forward(data, True, 4)
# for rep in enc_reps:
# 	print(rep.size(), torch.max(rep), torch.min(rep))

# print("decoded representations")
# for rep in dec_reps:
# 	print(rep.size(), torch.max(rep), torch.min(rep))

# criterion = nn.MSELoss()
# params = list(model.encoders[1].parameters()) + list(model.decoders[1].parameters())
# optimizer = optim.Adam(params)
# loss = level_loss(enc_reps, dec_reps, criterion)
# loss.backward()
# optimizer.step()
########### Test full model ################

def save_model(model, optimizer, level, epoch, save_file_path, min_criterion_vals):
	states = {
		'level': level,
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
		'criterion': min_criterion_vals,
		'config': params_dict
	}
	torch.save(states, save_file_path)

def update_min(min_vals, vals, key):
	updated = False
	min_val = min_vals[key]
	val = vals[key][1]
	if (min_val == None or val < min_val):
		min_vals[key] = val
		updated = True
	return updated

# Return:
# 1. the minimum loss between input and output computed on validation dataset
# 2. The best epoch ecountered (best in the sense validation loss is minimum)
def train_many_epochs(model, epochs, train_loader, valid_loader, test_set, level, optimizer, 
	scheduler, start_epoch):
	best_epoch = None

	min_criterion_vals = dict()
	for loss_name in LOSS_NAMES: min_criterion_vals[loss_name] = None

	weights_path = os.path.join(CONFIG_DIR, 'weights.pt')
	best_yet = False # Flag indicating whether current epoch achieves best results so far.
	patience = 5
	not_decreasing = 0

	criterion_vals, loss_for_cmpr, eval_epoch_time = evaluate_epoch(model, valid_loader, level)
	for epoch in range(start_epoch, epochs):
		print("Start epoch", epoch)
		train_loss, train_epoch_time = train_epoch(model, train_loader, level, optimizer)
		criterion_vals, loss_for_cmpr, eval_epoch_time = evaluate_epoch(model, valid_loader, level)
		
		print("  Epoch: %d, Train Loss: %.6f " %(epoch, train_loss), end='')

		metrics_dict = dict()
		for criterion_name in criterion_vals:
			vals = criterion_vals[criterion_name]
			name0 = 'val_' + criterion_name
			name1 = 'val_' + criterion_name + '_e2e'
			metrics_dict[name0] = vals[0]
			metrics_dict[name1] = vals[1]

		for metric_name in metric_names:
			print("%s: %.6f "%(metric_name, metrics_dict[metric_name]), end='')
		print('Time: %.2f, %.2f'%(train_epoch_time, eval_epoch_time))

		if (STORE_RESULTS):
			for loss_name in min_criterion_vals:
				updated = update_min(min_criterion_vals, criterion_vals, loss_name)
				if (loss_name == FINAL_CRTR and updated):
					# Store best results as well as model weights
					best_epoch = epoch
					best_yet = True
					save_model(model, optimizer, level, epoch, weights_path, min_criterion_vals)

					# test(model, test_set, 'pred.wav')
			lr = optimizer.param_groups[0]['lr']
			if (not best_yet):
				not_decreasing += 1
				if (not_decreasing >= patience):
					print("Early Stopping")
					break
				else:
					scheduler.step()
					print('Update learning rate to be', optimizer.param_groups[0]['lr'])

			# test(model, test_set, 'pred%d.wav'%epoch)
			# Store latest model weights
			save_model(model, optimizer, level, epoch, LAST_WEIGTHS_PATH, min_criterion_vals)

			info = {
			'level': level,
			'epoch': epoch,
			'training_loss': train_loss,
			'best_yet': 'better' if best_yet else ''
			}
			info.update(metrics_dict)

			write_epoch_result(info)
			pickle.dump({'epoch': epoch, 'level': level}, open(STATUS_PATH, 'wb'))

			best_yet = False  # Reset flag for next epoch

	return min_criterion_vals, best_epoch

def init_optimizer(model, level):
	print('Initialize optimizer at level', level)
	if (level == None):
		return optim.Adam(model.parameters(), lr=LR)
	else:
		level_index = level - 1
		params = list(model.encoders[level_index].parameters()) + list(model.decoders[level_index].parameters()) 
		return optim.Adam(params, lr=LR)

def train(model, optimizer, scheduler, train_loader, valid_loader, test_set, 
	start_level, start_epoch):
	start = time.time()

	print("Train Dataset", len(train_loader.dataset))
	print("Valid Dataset", len(valid_loader.dataset))
	print("Test Dataset", test_set[0].shape)

	best_epochs = []

	if (STORE_RESULTS and (not os.path.exists(metrics_filename))):
		# Write headers of csv file
		with open(metrics_filename, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=metrics_fieldnames)
			writer.writeheader()
	
	level = None # Level is ignored in non-greedy mode
	min_criterion_vals, best_epoch = train_many_epochs(model, N_EPOCHS, train_loader, 
		valid_loader, test_set, None, optimizer, scheduler, start_epoch)

	time_elapsed = time.time() - start
	print("Training time: %f"%time_elapsed)
	print("#####################################")

	return min_criterion_vals, best_epoch, best_epochs

################# Representation Learning of Utterances ###################################

def get_utters(video_data_path):
	slice_dict = dict() # Store the slicing point for each utterance
	# Structure: seg_id->(start_index, end_index)
	video_spect = []
	# Store learned representations in hdf5 files, as 
	with h5py.File(video_data_path, 'r') as f:
		# video_spect = [myReshape(f[name][:], time_steps) for name in f]
		prev_tot_len = 0
		sorted_seg_ids = sorted([int(seg_id) for seg_id in f])
		for seg_id in sorted_seg_ids:
			# print("seg_id", seg_id)
			seg_group = f[str(seg_id)]

			spectrogram = seg_group[:] 
			# [total_time_steps, input_dim]
			spectrogram = myReshape(spectrogram, TIME_STEPS)
			# [-1, fixed_time_steps, input_dim]
			shape = spectrogram.shape
			curr_len = shape[0]

			video_spect.append(spectrogram)
			slice_dict[int(seg_id)] = (prev_tot_len, prev_tot_len+curr_len)
			prev_tot_len += curr_len

	data = np.concatenate(video_spect)
	# [-1, fixed_time_steps(say, 128), INPUT_DIM]
	# print(data.shape, end='->')
	data = myReshape(data, BATCH_SIZE)
	# [N_BATCHES, BATCH_SIZE, TIME_STEPS, INPUT_DIM]
	print(data.shape)

	data, _, _ = normalize(data)
	data = torch.tensor(data, dtype=torch.float, device=DEVICE)

	return data, slice_dict

# utters: [N_BATCHES, BATCH_SIZE, TIME_STEPS, INPUT_DIM]
def get_repr_frm_model(model, utters):

	model.eval()

	repr_list = []
	for batch in utters:
		enc_reps, dec_reps = model.part_forward(batch, False, N_LEVELS)
		# enc_reps: list of [batch_size, ?, ?]
		representation = enc_reps[-1] # [batch_size, 1, repr_size]
		representation = torch.squeeze(representation, 1) # [batch_size, repr_size]
		repr_list.append(representation.detach().cpu().numpy())

	repr_array = np.array(repr_list)
	# [N_BATCHES, BATCH_SIZE, RPER_SIZE]
	return repr_array

# reprs: [N_BATCHES, BATCH_SIZE, RPER_SIZE] np array
def store_repr(reprs, slice_dict, repr_store_path):
	reprs = reprs.reshape(-1, reprs.shape[-1]) # [-1, REPR_SIZE]
	with h5py.File(repr_store_path, 'w') as f:
		for seg_id in slice_dict:
			# print('seg_id', seg_id)
			(si, ei) = slice_dict[seg_id]
			seg_repr = reprs[si:ei]
			seg_dset = f.create_dataset("%d"%seg_id, data=seg_repr)

def rep_learning(model, repr_store_path):
	model.eval()
	frame_size = INPUT_DIM / 2
	REP_DATA_PATH = "/media/bighdd7/irene/HLSTM/data/WAV_11025/"
	spect_utters_dir = REP_DATA_PATH + 'segmented/spectrograms/frame_%d/'%frame_size
	print(spect_utters_dir)
	
	if (not os.path.exists(repr_store_path)): os.makedirs(repr_store_path)

	file_count = 0

	for file in os.listdir(spect_utters_dir):
		if (not file.endswith('.hdf5')): continue
		file_count += 1

		# Get video ID
		vid_ei = file.rfind('.')
		video_id = file[:vid_ei]
		# video_id = '2WGyTLYerpo'
		# print(video_id)

		utters, slice_dict = get_utters(spect_utters_dir + file)
		# utters: [N_BATCHES, BATCH_SIZE, TIME_STEPS, INPUT_DIM]

		reprs = get_repr_frm_model(model, utters)
		# reprs: [N_BATCHES, BATCH_SIZE, RPER_SIZE]
		
		dest_file_path = os.path.join(repr_store_path, video_id + '.hdf5')
		store_repr(reprs, slice_dict, dest_file_path)
		f = h5py.File(dest_file_path, 'r')
		# print(f.keys())

	print(file_count, "videos")

def load_truth():
	truth_path = '/media/bighdd5/Paul/mosi/Meta_data/boundaries_sentimentint_avg.csv'
	truth_dict = defaultdict(dict)
	with open(truth_path) as f:
		lines = f.read().split("\n")
	for line in lines:
		if line != '':
			line = line.split(",")
			video_id = line[2]
			seg_id = line[3]
			# print(video_id, type(video_id), seg_id, type(seg_id))
			# line: [start time, end time, video_id, segment_id, sentiment]
			truth_dict[line[2]][line[3]] = {'start_time': float(line[0]), 'end_time':float(line[1]), 'sentiment':float(line[4])}
	return truth_dict

def store_rep_for_mfn(fold):
	# train, valid, test = load_data("mosei", 400)
	ARCHITECTURE = "nocake"
	SPACE = 'space0'
	FEATURES_SIZE = 50
	TIME_STEPS = 256
	FREQ = 11025
	rep_size = FEATURES_SIZE * TIME_STEPS / FREQ

	# rep_data_path = '/media/bighdd7/irene/HLSTM/results/%s_results/test_space/config%d/repr/'%(ARCHITECTURE, REP_NUM)
	# new_path = '/media/bighdd7/irene/HLSTM/Memory-Fusion-Network/rep_data/config%d'%REP_NUM
	# config_dir = '/media/bighdd7/irene/HLSTM/results/%s_results/%s/config%d/'%(ARCHITECTURE, SPACE, REP_NUM)
	rep_data_path = os.path.join(CONFIG_DIR, 'repr/')
	new_path = os.path.join(CONFIG_DIR, 'detailed_repr')
	if (not os.path.exists(new_path)): os.makedirs(new_path)
	truth_dict = load_truth()

	covarep_dict = dict()
	for video_id in fold:
		# video_id = '2WGyTLYerpo'
		print(video_id)
		video_file_path = os.path.join(rep_data_path, video_id + '.hdf5')
		# f = h5py.File(video_file_path, 'r')
		# print(video_id, f.keys())
		# return
		new_video_file_path = os.path.join(new_path, video_id+'.hdf5')
		new_f = h5py.File(new_video_file_path, "w")

		video_dict = truth_dict[video_id]
		covarep_vid = dict()
		with h5py.File(video_file_path, 'r') as f:
			seg_id = 1
			if (video_id == '2WGyTLYerpo'): print(video_id, f.keys())
			while (str(seg_id) in f):
				seg_grp = new_f.create_group(str(seg_id))

				seg_dict = video_dict[str(seg_id)]
				rep = f[str(seg_id)][:]
				estimate_time = rep.shape[0] * FEATURES_SIZE * TIME_STEPS / FREQ
				true_time = seg_dict['end_time'] - seg_dict['start_time']
				# print(seg_id, rep.shape, estimate_time, true_time)
				covarep_seg = dict()
				for i in range(rep.shape[0]):
					rep_grp = seg_grp.create_group(str(i))
					rep_grp.create_dataset('data', data=rep[i])
					rep_grp.attrs['rep_id'] = i
					rep_grp.attrs['start_time_rep'] = i*rep_size
					rep_grp.attrs['end_time_rep'] = (i+1)*rep_size
					covarep_seg[i] = {"data": rep[i], "rep_id": i, 
						"start_time_rep": i*rep_size, "end_time_rep": (i+1)*rep_size}
				covarep_vid[seg_id] = covarep_seg
				seg_id += 1	

################# Representation Learning of Utterances ###################################

def main():

	model = enc_dec(params_dict, DEVICE)
	model = model.to(DEVICE)

 ###################### Debugging #############################
	# optimizer = optim.Adam(model.parameters(), lr=LR)
	# optimizer.zero_grad()

	# data = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_DIM).to(DEVICE)
	# wt_batch = torch.randn(BATCH_SIZE, TIME_STEPS, INPUT_DIM).to(DEVICE)

	# enc_reps, dec_reps = model(data, False)
	# print("Encoded: ")
	# for rep in enc_reps: 
	# 	print(rep.shape, torch.max(rep).item(), torch.min(rep).item())
	# print("Decoded: ")
	# for i in range(len(dec_reps)):
	# 	print(i)
	# 	for rep in dec_reps[i]:
	# 		print(rep.shape, torch.max(rep).item(), torch.min(rep).item())

	# loss_dict = all_losses(enc_reps, dec_reps[-1], True, True)
	# for key in loss_dict: 
	# 	print(key, loss_dict[key].item())
	# loss = loss_dict[FINAL_CRTR]

	# loss.backward()
	# optimizer.step()
	# return

	# assert(False)
 ###################### Debugging #############################

	if (TRAIN_MODE):
		start = time.time()
		################## Prepare Data ######################
		print('Prepare data...')

		test_set = load_test_data(INPUT_DIM, TIME_STEPS)

		train_loader = get_data_loader('train_length.csv', INPUT_DIM, 
			TIME_STEPS, BATCH_SIZE, True)
		valid_loader = get_data_loader('valid_length.csv', INPUT_DIM, 
			TIME_STEPS, BATCH_SIZE, False)

		time_elapsed = time.time() - start 
		print("Getting data takes %fs."%time_elapsed)
		print("#####################################")
		print("Training...")
		if (RESUME and os.path.exists(LAST_WEIGTHS_PATH)):

			states = torch.load(LAST_WEIGTHS_PATH)
			start_epoch = states['epoch']
			start_level = states['level']

			model.load_state_dict(states['state_dict'])
			optimizer = init_optimizer(model, start_level)
			optimizer.load_state_dict(states['optimizer'])

			print('Resume from level', start_level, end=' ')
			print('epoch %d'%start_epoch)
		else:
			print("Start fresh run.")
			start_level = None
			start_epoch = 0
			optimizer = init_optimizer(model, start_level)
			if (INIT_WEIGHTS != None):
				print('Initial weights: ', INIT_WEIGHTS)
				states = torch.load(INIT_WEIGHTS)
				model.load_state_dict(states['state_dict'])
				optimizer.load_state_dict(states['optimizer'])

		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
		min_criterion_vals, best_epoch, best_epochs = train(model, optimizer, scheduler, 
			train_loader, valid_loader, test_set, start_level, start_epoch)

		# Save results in a csv file
		if (STORE_RESULTS):
			write_results(CSVFILEPATH, min_criterion_vals, best_epoch, best_epochs)

	else:
		if (WEIGHTS_PATH != None and os.path.exists(WEIGHTS_PATH)):
			print('Reconstruct/Represenation Learning by using model weights at:')
			print(WEIGHTS_PATH)
			states = torch.load(WEIGHTS_PATH)
			model.load_state_dict(states['state_dict'])
		else:
			print('Caution: Weights path does not exist', WEIGHTS_PATH)
			return

		if (TEST_MODE):
			print("Reconstructing...")
			test_set = load_test_data(INPUT_DIM, TIME_STEPS)
			test(model, test_set, 'pred.wav')
		if (LEARN_MODE):
			repr_store_path = CONFIG_DIR + 'repr/'
			rep_learning(model, repr_store_path)
			for fold in [train_fold, valid_fold, test_fold]:
				store_rep_for_mfn(fold)
			
if __name__ == "__main__":
	main()


'''
1. Change settings: Done
2. Save grid search results (see previous results): YEAH IT SAVES RESULTS.
3. Change loss function: Done
4. When saving weights, save config and metric with it: Done
5. Change method of doing grid search (use dictionary): Don't change it.
6. Check representation learning works
7. Early stopping

*: Check difference between repr and new_repr: It's different! but I don't know why...

Steps:
1. Run representation learning
2. Run myrep_loader.py 

'''



