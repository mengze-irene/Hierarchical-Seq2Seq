from __future__ import division, print_function
import os
import itertools
import time
import csv
import pickle
import random
import argparse
from settings import ARCH_DIR, GS_FIELDNAMES, PARAM_LIST_FILENAME, CSVFILENAME, PARAM_NAMES

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--index", type=int)
parser.add_argument("--init", action="store_true")
parser.add_argument("--concat", action="store_true")
parser.add_argument('--assign', nargs='+', type=int)
parser.add_argument("--space_id", type=int)
parser.add_argument("-p", "--pseudo", action="store_true")
args = parser.parse_args()

index = args.index
INIT = args.init
CONCAT = args.concat
ASSIGNED_CONFIGS = args.assign
SPACE_ID = args.space_id
PARENT_DIR = ARCH_DIR + 'space%d/'%SPACE_ID
pseudo = args.pseudo

PARAM_LIST_FILEPATH = PARENT_DIR + PARAM_LIST_FILENAME
CSVFILEPATH = PARENT_DIR + CSVFILENAME

#################### Parameters #########################

INPUT_DIM_LIST = [100]
TIME_STEPS_LIST = [256]
HIDDEN_DIMS_LIST = [[512, 512, 512]]
# HIDDEN_DIMS_LIST = [[32, 24, 16]]
N_SPLITS_LIST = [[32, 8, 1]]
# LR_LIST = [0.0001, 0.0005, 0.00001]
LR_LIST = [0.01, 0.001, 0.0005]
BATCH_SIZES = [512]
GREEDY_OPTIONS = [False]
REVERSE_OPTIONS = [True]
DROPOUT_RATES = [0.5, 0]
LOSS_WEIGHTS = [[1, 0, 0, 0], [1000, 0, 1, 0]]
N_EPOCHS = 50
#################### Parameters #########################


################# Helper Functions ######################

def is_valid_params(params):
	time_steps = params[1]
	hidden_dims = params[2]
	n_splits = params[3]

	assert(len(hidden_dims) == len(n_splits))
	prev_split = time_steps
	for split in n_splits:
		if(prev_split % split != 0): 
			print(params)
			assert(False)
		prev_split = split

def paramlist2dict(params):
	input_dim = params[0]
	time_steps = params[1]
	hidden_dims = params[2]
	n_splits = params[3]
	lr = params[4]
	batch_size = params[5]
	greedy = params[6]
	reverse = params[7]
	dropout = params[8]
	loss_weights = params[9]
	n_levels = len(hidden_dims)

	config = {'input_dim': input_dim, 'time_steps': time_steps, 'hidden_dims': hidden_dims, 
		'n_splits': n_splits, 'lr': lr, 'batch_size': batch_size, 
		'reverse': reverse, 'dropout': dropout, 'loss_weights': loss_weights,
		'n_levels': n_levels}
	return config

# Train with set of hyperparameters specified by i.
def train_with_hp(params, i):
	print('CONFIGURATION %d'%i)

	config_dir = PARENT_DIR+'config%d/'%i
	print(config_dir)

	input_dim = params[0]
	time_steps = params[1]
	hidden_dims = params[2]
	n_splits = params[3]
	lr = params[4]
	batch_size = params[5]
	greedy = params[6]
	reverse = params[7]
	dropout = params[8]
	loss_weights = params[9]
	n_levels = len(hidden_dims)

	resume_flags = ''
	status_path = config_dir + 'status.p'
	if (os.path.exists(config_dir)): 
		# Check status of the current run: whether it's unfinished
		if (os.path.exists(status_path)):
			status_file = open(status_path, 'rb')
			status = pickle.load(status_file)
			resume_epoch = status['epoch']
			resume_level = status['level']

			if ((not greedy and (resume_epoch >= N_EPOCHS-1)) 
				or (greedy and ((resume_level > n_levels) 
					or (resume_level == n_levels) and (resume_epoch >= N_EPOCHS-1)))):
				print('Run already finished')
				return
			print('Resuming run from level %s and epoch %d'
				%('%d'%resume_level if (resume_level!=None) else 'N/A', resume_epoch))
			resume_flags = ' --resume '
			if (resume_level != None):
				resume_flags += '--resume_level %d '%resume_level
	else: 
		os.makedirs(config_dir)

	list_format = '%d '*n_levels
	# greedy_flag = ' -g ' if greedy else ''
	reverse_flag = ' -r ' if reverse else ''
	pseudo_flag = ' -p ' if pseudo else ''

	command = ('python main.py '
		+ ' --input_dim %d '%input_dim
		+ ' --time_steps %d '%time_steps
	    + ' --hidden_dims ' + list_format%(tuple(hidden_dims))
		+ ' --n_splits '+list_format %(tuple(n_splits)) 
		+ ' --learning_rate %f '%lr
		+ ' --batch_size %d '%batch_size
		+ reverse_flag 
		+ ' --dropout %f '%dropout
		+ ' --loss_weights ' + ('%d '*len(loss_weights))%(tuple(loss_weights))
		+ ' --config %d'%i
		+ resume_flags
		+ ' --epochs %d '%N_EPOCHS
		+ ' --parent_dir %s '%PARENT_DIR
		+ pseudo_flag)
	os.system(command)

def get_interval(first_index, last_index, chunks):
	size = last_index - first_index
	subsize = size // chunks
	start = index * subsize + first_index
	end = start + subsize

	return start, end

def init_param_space():
	# Read previous configurations
	old_param_list = []
	if (os.path.exists(PARAM_LIST_FILEPATH)):
		param_file = open(PARAM_LIST_FILEPATH, 'rb')
		old_param_list = pickle.load(param_file)
		# old_param_list = old_param_list[:24]
		param_file.close()
		print("Size of old parameter space: %d"%len(old_param_list))

	param_list = list(itertools.product(INPUT_DIM_LIST, TIME_STEPS_LIST, HIDDEN_DIMS_LIST,
		N_SPLITS_LIST, LR_LIST, BATCH_SIZES, GREEDY_OPTIONS, REVERSE_OPTIONS, DROPOUT_RATES, LOSS_WEIGHTS))
	print("Size of new parameter space: %d"%len(param_list))
	random.shuffle(param_list)
	# Check if each set of hyperparameters is valid
	for params in param_list: is_valid_params(params)

	# Concatenate old and new parameter lists
	if (CONCAT):
		param_list += old_param_list
	print("Size of total parameter space: %d"%len(param_list))
	print('Space %d'%SPACE_ID)
	pickle.dump(param_list, open(PARAM_LIST_FILEPATH, 'wb'))

	csvfilepath = os.path.join(PARENT_DIR, 'params.csv')
	with open(csvfilepath, 'at') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=PARAM_NAMES)
		writer.writeheader()
		for i in range(len(param_list)):
			params = param_list[i]
			content = paramlist2dict(params)
			content['config_num'] = i
			writer.writerow(content)

def load_param_space():
	param_file = open(PARAM_LIST_FILEPATH, 'rb')
	param_list = pickle.load(param_file)
	param_file.close()
	print("Size of parameter space: %d"%len(param_list))

	return param_list

def write_csv_header():
	if (not os.path.exists(CSVFILEPATH)):
		with open(CSVFILEPATH, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=GS_FIELDNAMES)
			writer.writeheader()

################# Helper Functions ######################

if __name__ == "__main__":
	if (not os.path.exists(PARENT_DIR)): os.makedirs(PARENT_DIR)

	if (INIT):
		init_param_space()
		write_csv_header()
	else:
		param_list = load_param_space()

		if (ASSIGNED_CONFIGS != None):
			for config in ASSIGNED_CONFIGS:
				train_with_hp(param_list[config], config)
		else:
			for i in range(len(param_list)):
				train_with_hp(param_list[i], i)


