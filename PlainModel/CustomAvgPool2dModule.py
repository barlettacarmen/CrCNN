import torch
import torch.nn as nn
from CustomAvgPool2d import CustomAvgPoolingFunction

class CustomAvgPooling(nn.Module):
	def __init__(self):
		super(CustomAvgPooling,self).__init__()
	def forward(self,input):
		return CustomAvgPoolingFunction.apply(input)