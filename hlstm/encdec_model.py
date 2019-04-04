from __future__ import division, print_function
import numpy as np
import os
from collections import defaultdict
import time
import torch
import torch.nn as nn
import torch.optim as optim
from speechLSTM import encoderLayer, decoderLayer

class enc_dec(nn.Module):
	# n_levels = number of encoders/decoders
	# hidden_dims is a list of hidden dimenstions
	def __init__(self, config, device):

		super(enc_dec, self).__init__()

		self.input_dim = config['input_dim']
		self.n_levels = config['n_levels']
		self.hidden_dims = config['hidden_dims']
		self.n_splits = config['n_splits']
		self.reverse = config['reverse']
		self.dropout = config['dropout']
		self.device = device

		encoder_list = []
		decoder_list = []
		self.enc_reps = None

		for i in range(self.n_levels):
			curr_input_dim = self.input_dim if (i == 0) else self.hidden_dims[i-1]
			dec_output_dim = curr_input_dim
			hidden_dim = self.hidden_dims[i]
			n_split = self.n_splits[i]
			encoder = encoderLayer(n_split, curr_input_dim, hidden_dim)
			decoder = decoderLayer(n_split, curr_input_dim, hidden_dim, dec_output_dim, device)

			encoder_list.append(encoder)
			decoder_list.append(decoder)

		self.dropout = nn.Dropout(self.dropout)
		self.encoders = nn.ModuleList(encoder_list)
		self.decoders = nn.ModuleList(decoder_list)

    # Utility function: reverse tensor x in dimension given by dim
	def flip_tensor(self, x, dim):
		steps = x.size()[dim]
		indices = torch.tensor(np.arange(steps-1, -1, -1))
		indices = indices.to(self.device)
		return torch.index_select(x, dim, indices)

	def normalize(self, x):
		time_steps = x.size()[1]

		fc_input = torch.mean(x, 1)
		multiplier = self.fc(fc_input)
		multiplier = torch.unsqueeze(multiplier, 1)
		multiplier = multiplier.expand(self.batch_size, time_steps, self.input_dim)
		self.multiplier = multiplier

		encdec_input = x * self.multiplier

		return encdec_input

	def denormalize(self, x):
		return torch.div(x, self.multiplier)

	def encode(self, x, end_level):
	
		enc_reps = [x]
		rep = x

		if (self.reverse):
			rep = self.flip_tensor(rep, 1) # Reverse input in time steps.

		for i in range(end_level):
			# Dropout before inputting representation into encoders
			rep = self.dropout(rep)

			encoder = self.encoders[i]
			hiddens = encoder(rep)
			# hiddens/cells: [batch_size, n_split, hidden_dim]
			rep = hiddens
			# rep: [batch_size, n_split, 2*hidden_dim]
			if (self.reverse):
				enc_reps.append(self.flip_tensor(rep, 1))
			else:
				enc_reps.append(rep)

		return enc_reps


	# Given encoded states at start_level, return decoded representations
	# of all lower levels (including start_level)
	def decode(self, states, start_level, teacher_forcing, total_time_steps, enc_reps):
		assert(start_level <= self.n_levels)

		# if (teacher_forcing): assert(enc_reps != None)

		# Discard the raw input if perform self norm.

		dec_reps_i = []
		for j in range(start_level-1, -1, -1):
			decoder = self.decoders[j]
			n_split = self.n_splits[j]

			# Obtain input, initial hidden and cell state
			last_dim = states.size()[-1]
			hidden = states[:, :, :]
			# hidden/cell: [batch_size, n_split, hidden_dim]
			target_dim = self.hidden_dims[j-1] if (j > 0) else self.input_dim
			next_time_steps = self.n_splits[j-1] if (j > 0) else total_time_steps
			sub_time_steps = next_time_steps // n_split 
			start_input = torch.zeros((self.batch_size * n_split, 1, target_dim), dtype=torch.float)
			start_input = start_input.to(self.device)
			if (teacher_forcing):
				target = enc_reps[j]
				assert(target_dim == target.size()[-1])
				assert(next_time_steps == target.size()[1])
				# target = [batch_size, next n_split, 2 * next hidden dim]
				# Push target one time step back by padding zeros in the front
				target = target.view(self.batch_size * n_split, -1, target_dim)
				target = torch.cat((start_input, target[:, :-1, :]), 1)
				target = target.view(self.batch_size, -1, target_dim)
				inputs = target

			else:
				inputs = start_input
				inputs = inputs.view(self.batch_size, -1, target_dim)

			# Decode
			inputs = self.dropout(inputs)
			states = decoder(inputs, hidden, teacher_forcing, sub_time_steps)
			# print("level=%d sub=%d split=%d"%(j, sub_time_steps, n_split), states.size())
			# states: [batch_size, next n_split, 2*next hidden dim]
			dec_reps_i.append(states)

		dec_reps_i.reverse()
		# length = start_level (1-indexed)
		return dec_reps_i

	def part_forward(self, x, teacher_forcing, level):
		# x = [batch_size, time steps, input dim]
		self.batch_size = x.shape[0]
		time_steps = x.shape[1]

		enc_reps = self.encode(x, level)
		states = enc_reps[level]
		dec_reps = self.decode(states, level, teacher_forcing, time_steps, enc_reps)

		return enc_reps, dec_reps

	def forward(self, x, teacher_forcing):
		# x = [batch_size, time steps, input dim]
		self.batch_size = x.shape[0]
		time_steps = x.shape[1]

		enc_reps = self.encode(x, self.n_levels)

		dec_reps = []
		for i in range(self.n_levels, 0, -1):
			states = enc_reps[i]
			dec_reps_i = self.decode(states, i, teacher_forcing, time_steps, enc_reps)
			dec_reps.append(dec_reps_i)
		dec_reps.reverse()
		return enc_reps, dec_reps

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.lstm = nn.LSTM(100, 128, batch_first=True)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def test_model():
	net = Net()
	net = net.to('cuda')
	return

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config = {'input_dim': 100, 'n_levels': 3, 'hidden_dims': [128, 64, 24],
		'n_splits': [32, 8, 1], 'reverse': True, 'dropout': 0.5}
	model = enc_dec(config, device)
	model = model.to(device)

if __name__ == '__main__':
	test_model()
