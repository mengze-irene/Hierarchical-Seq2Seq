# This file is not in use.
# Might be changed later.

import matplotlib  
matplotlib.use('Agg')   
import matplotlib.pyplot as plt  
import torch
import os
import csv
import numpy as np
import argparse
# Deprecated
from settings import PARENT_DIR

def plot_losses(config_num, metric_names):
	directory = parent_dir + 'config%d/'%config_num
	csvfilename = directory + 'metrics.csv'
	with open(csvfilename, 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		next(reader) # Skip header
		data = [r for r in reader]

	n_points = len(data)
	x = np.arange(n_points)
	
	for i in range(len(metric_names)):
		metric_name = metric_names[i]
		history = []
		for r in data:
			history.append(r[metric_name])
		history = np.array(history)

		plt.plot(x, history, label=metric_name)

	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.xlabel('Epochs')
	plt.savefig(directory+'loss.png', bbox_inches='tight')

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', type=int)
	args = parser.parse_args()

	metric_names = ['training_loss','valid_loss','valid_loss_end2end']

	config_num = args.config
	plot_losses(config_num, metric_names)


