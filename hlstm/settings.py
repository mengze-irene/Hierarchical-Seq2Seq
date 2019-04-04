import torch

ARCH_DIR = '/media/bighdd7/irene/HLSTM/results/nocake_results/'
PARAM_NAMES = ['config_num', 'reverse', 'batch_size',
			'input_dim', 'time_steps', 'n_levels', 'hidden_dims', 'n_splits', 
			  'lr', 'dropout', 'loss_weights']

LOSS_NAMES = ['mse', 'mae', 'corr', 'lap', 'hybrid']
FINAL_CRTR = 'hybrid' # The criterion used for training and comparison between different runs
GS_FIELDNAMES = PARAM_NAMES + ['best_epochs', 'best_epoch'] + LOSS_NAMES
PARAM_LIST_FILENAME = 'params.p'
CSVFILENAME = 'results.csv'


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
# device = 'cpu'