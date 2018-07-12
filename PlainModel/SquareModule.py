import torch
import torch.nn as nn
from Square import SquareFunction

class Square(nn.Module):
	def __init__(self):
		super(Square,self).__init__()
	def forward(self,input):
		return SquareFunction.apply(input)