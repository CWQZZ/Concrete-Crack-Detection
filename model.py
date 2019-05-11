import torch
import torch.nn as nn
import pdb


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv_layers = nn.Sequential(
						nn.Conv2d(3, 24, kernel_size=6, stride=2, padding=(1,0), dilation=2), # output feature size: 61
						nn.LeakyReLU(0.01),
						nn.BatchNorm2d(24),
						nn.Conv2d(24, 48, kernel_size=3, stride=2,padding=(1,0), dilation=2), # output feature size: 29 
						nn.LeakyReLU(0.01),
						nn.BatchNorm2d(48),
						nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1), # output feature size: 11
						nn.LeakyReLU(0.01),
						nn.BatchNorm2d(96),
						nn.Conv2d(96, 192, kernel_size=4, stride=2, padding=0),
						nn.LeakyReLU(0.01),
						nn.Conv2d(192, 192 * 2, kernel_size=6, stride=1, padding=0),
						nn.LeakyReLU(0.01))
		
		self.linear_feats = nn.Sequential(nn.Linear(192*2, 192*4),
									 nn.LeakyReLU(0.01),
									 nn.Linear(192*4, 2))

	def forward(self, x):
		x_feats = self.conv_layers(x)
		x_flatten = x_feats.view(x.size(0), -1)

		return self.linear_feats(x_flatten)

