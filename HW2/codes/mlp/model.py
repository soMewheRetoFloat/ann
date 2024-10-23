# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm1d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum=0.1, eps=1e-5, gamma=1, beta=0):
		super(BatchNorm1d, self).__init__()
		self.num_features = num_features

		# Parameters
		# f = W @ x + b
		self.weight = Parameter(torch.empty(num_features))
		self.bias = Parameter(torch.empty(num_features))

		# Store the average mean and variance
		# !!! input of each mini_batch has mean of 0 and var of 1
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))
		
		# Initialize your parameter
		init.ones_(self.weight)
		init.zeros_(self.bias)

		# store other params
		self.momentum = momentum
		self.norm_eps = eps
  
	def forward(self, input: torch.Tensor):
		# input: [batch_size, num_feature_map * height * width]
		# input of each mini_batch has mean of 0 and var of 1

		# ref https://blog.csdn.net/chen_kl86/article/details/131389696

		if self.training:
			# [num_feature_map * height * width]
			observed_mean = torch.mean(input, dim=0)
			observed_var = torch.var(input, dim=0, unbiased=False)
			self.running_mean = self.momentum * observed_mean + (1 - self.momentum) * self.running_mean
			self.running_var = self.momentum * observed_var + (1 - self.momentum) * self.running_var
		else:
			observed_mean = self.running_mean
			observed_var = self.running_var
   
		# normalize
		norm_initial = (input - observed_mean) / torch.sqrt(observed_var + self.norm_eps)
		norm_extend = self.weight * norm_initial + self.bias
  
		return norm_extend
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		assert 0 <= p <= 1
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map * height * width]
  
		# weighted by a factor 1 - p
		q = 1 - self.p
		
		if self.training:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			# 根据input的 shape 和 p 生成一个 mask, 1 出现的概率为 1 - p
			mask = (torch.bernoulli(torch.full_like(input, q)).float().to(device))
			input = input * mask / q
		return input
	# TODO END

class Model(nn.Module):
	def __init__(self, drop_rate=0.5):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here

		# from dataset
		input_dim = 3072
		hidden_dim = 1024
		class_num = 10
  
		# input - Linear – BN – ReLU – Dropout – Linear – loss
		self.sequence_model = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			# BatchNorm1d(hidden_dim),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.Linear(hidden_dim, class_num)
		)
		# TODO END
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x, y=None):
		# TODO START
		# the 10-class prediction output is named as "logits"
		logits = self.sequence_model(x)
		# TODO END

		pred = torch.argmax(logits, 1)  # Calculate the prediction result
		if y is None:
			return pred
		loss = self.loss(logits, y)
		correct_pred = (pred.int() == y.int())
		acc = torch.mean(correct_pred.float())  # Calculate the accuracy in this mini-batch

		return loss, acc
