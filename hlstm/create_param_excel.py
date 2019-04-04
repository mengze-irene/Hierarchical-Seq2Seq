import csv
import pickle
from settings import ARCH_DIR, PARAM_NAMES, PARAM_LIST_FILENAME
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--space_id", '-si', type=int)
args = parser.parse_args()
SPACE_ID = args.space_id

PARENT_DIR = ARCH_DIR
if (SPACE_ID != None): PARENT_DIR += 'space%d/'%SPACE_ID
PARAM_LIST_FILEPATH = PARENT_DIR + PARAM_LIST_FILENAME

def write_results(csvfilename, params_dict):
	marker = '.'
	with open(csvfilename, 'at') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=PARAM_NAMES)
		contents = dict()
		for fieldname in PARAM_NAMES:
			val = params_dict[fieldname]
			contents[fieldname] = val
		writer.writerow(contents)

def main():
	param_file = open(PARAM_LIST_FILEPATH, 'r')
	param_list = pickle.load(param_file)
	param_file.close()

	csvfilepath = PARENT_DIR + 'params.csv'
	with open(csvfilepath, 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=PARAM_NAMES)
		writer.writeheader()

	for CONFIG_NUM in range(len(param_list)):
		params = param_list[CONFIG_NUM]

		INPUT_DIM = params[0]
		TIME_STEPS = params[1]
		HIDDEN_DIMS = params[2]
		N_SPLITS = params[3]
		LR = params[4]
		BATCH_SIZE = params[5]
		GREEDY = params[6]
		REVERSE = params[7]
		DROPOUT = params[8]
		LOSS_WEIGHTS = params[9]
		N_LEVELS = len(HIDDEN_DIMS)


		params_dict = {'config_num' : CONFIG_NUM, 
						'input_dim': INPUT_DIM, 'time_steps': TIME_STEPS, 'n_levels': N_LEVELS,
		                'hidden_dims': HIDDEN_DIMS, 'n_splits': N_SPLITS, 'lr': LR, 'greedy' : GREEDY,
		                'reverse' : REVERSE, 'batch_size' : BATCH_SIZE, 'loss_weights': LOSS_WEIGHTS, 
		                'dropout' : DROPOUT}

		write_results(csvfilepath, params_dict)


if __name__ == "__main__":
	main()

