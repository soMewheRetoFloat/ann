# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
class BatchNorm2d(nn.Module):
	# TODO START
	def __init__(self, num_features, momentum=0.1, eps=1e-5, gamma=1, beta=0):
		super(BatchNorm2d, self).__init__()
		self.num_features = num_features

		# Parameters
		# gamma
		self.weight = Parameter(torch.empty(num_features))
		# beta
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

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		if self.training:
			# [num_feature_map * height * width]
			observed_mean = torch.mean(input, dim=(0, 2, 3))
			observed_var = torch.var(input, dim=(0, 2, 3), unbiased=False)
			self.running_mean = self.momentum * observed_mean + (1 - self.momentum) * self.running_mean
			self.running_var = self.momentum * observed_var + (1 - self.momentum) * self.running_var
		else:
			observed_mean = self.running_mean
			observed_var = self.running_var
   
		# normalize
		# N, H, W squeezed by mean and var
		N, C, H, W = input.shape
		norm_initial = (input - observed_mean.view(1, C, 1, 1)) / torch.sqrt(observed_var.view(1, C, 1, 1) + self.norm_eps)
		norm_extend = self.weight.view(1, C, 1, 1) * norm_initial + self.bias.view(1, C, 1, 1)
  
		return norm_extend
	# TODO END

class Dropout(nn.Module):
	# TODO START
	def __init__(self, p=0.5):
		super(Dropout, self).__init__()
		assert 0 <= p <= 1
		self.p = p

	def forward(self, input):
		# input: [batch_size, num_feature_map, height, width]
		N, C, H, W = input.shape
		q = 1 - self.p
		
		if self.training:
			# 2d 要直接 0 一整个channel 于是让单位变为channel
			shape_tensor = torch.zeros(size=(N, C, 1, 1))
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
			# 根据input的 shape 和 p 生成一个 mask, 1 出现的概率为 1 - p
			mask = (torch.bernoulli(torch.full_like(shape_tensor, q)).float().to(device))
			input = input * mask / q
		return input
	# TODO END
 
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, drop_rate=0.5, oper_seed=1):
        super(ConvBlock, self).__init__()
        operations = [
			nn.Conv2d
   			(in_channels=in_channel, 
             out_channels=out_channel, 
             kernel_size=3,
             stride=1,
             padding=1),
			BatchNorm2d(out_channel),
			nn.ReLU(),
			Dropout(drop_rate),
			nn.MaxPool2d(kernel_size=2, stride=2),
		]
        ordered_operations = []
        if oper_seed == 1: # 正常顺序
            ordered_operations = operations
        elif oper_seed == 2: # 调换 relu 和 norm 的位置
            ordered_operations = [operations[0], operations[2], operations[1], operations[3], operations[4]]
        elif oper_seed == 3: # 去掉 dropout
            ordered_operations = [operations[0], operations[1], operations[2], operations[4]]
        elif oper_seed == 4: # 调换 maxpool 到最前面
            ordered_operations = [operations[4], operations[0], operations[1], operations[2], operations[3]]
        else:
            ordered_operations = operations
            
        # [conv, bn, relu, dropout, maxpool]
        self.conv_block = nn.Sequential(*ordered_operations)
        
    def forward(self, x):
        return self.conv_block(x)
        
class Model(nn.Module):
	def __init__(self, drop_rate=0.5, conv_block_seed=1):
		super(Model, self).__init__()
		# TODO START
		# Define your layers here
  
		# from dataset
		in_channel = 3
		hidden_channel = 16
		out_channel = 32
		
		hidden_dim = 1024
		class_num = 10
  
		# input – Conv – BN – ReLU – Dropout – MaxPool 
  		# – Conv – BN – ReLU – Dropout – MaxPool
		# – Linear – loss
		self.sequence_model = nn.Sequential(
			ConvBlock(in_channel, hidden_channel, drop_rate, conv_block_seed),
			ConvBlock(hidden_channel, out_channel, drop_rate, conv_block_seed),
			# flat or gg
			nn.Flatten(),
			nn.Linear(32 * 8 * 8, class_num)
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
