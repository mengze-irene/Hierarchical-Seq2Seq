import pickle, os
import argparse
from settings import ARCH_DIR

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", nargs='+', type=int)
parser.add_argument("-l", "--level", type=int)
parser.add_argument("--last", action="store_true")
parser.add_argument("-p", "--pseudo", action="store_true")
parser.add_argument("-a", "--audio", action="store_true")
parser.add_argument("-r", "--repr", action="store_true")
parser.add_argument("-s", "--space_id", type=int)

args = parser.parse_args()
config_nums = args.config
level = args.level
last_weights = args.last
pseudo = args.pseudo 
pseudo_flag = ' -p ' if pseudo else ''
rec_audio = args.audio
learn_repr = args.repr
space_id = args.space_id
audio_flag = ' --test_mode ' if rec_audio else ''
repr_flag = ' --learn_mode ' if learn_repr else ''
# PARENT_DIR = ARCH_DIR  + 'test_space/'
PARENT_DIR = os.path.join(ARCH_DIR, 'space%d/'%space_id)

def test_config(config_num):
	directory = os.path.join(PARENT_DIR, 'config%d/'%config_num)
	hyperparams_path = os.path.join(directory, 'hyperparams.txt')
	file = open(hyperparams_path, 'r')
	command = file.read()
	file.close()

	# Temporary approach. To be changed later
	# omit_string = '--train_mode'
	# length = len(omit_string)
	# index = command.find(omit_string)
	# command = command[:index] + command[(index+length):]

	file = 'L%d-weights.pt'%level if (level != None) else 'weights.pt'
	if (last_weights):
		file = 'last_weights.pt'

	command += audio_flag + repr_flag  + pseudo_flag 
	command += ' --weights_file ' + file
	command += ' --no_command --no_training --learn_mode' 
	# command += ' --parent_dir %s'%PARENT_DIR
	print(command)
	os.system(command)

if __name__ == "__main__":
	for config_num in config_nums:
		test_config(config_num)
