import torch
import math
from torch.autograd import gradcheck,Variable

class SquareFunction(torch.autograd.Function):
	def __init__(self):
		super(SquareFunction, self).__init__()

	@staticmethod
	def forward(ctx,input):
		output=torch.pow(input,2)
		ctx.save_for_backward(input)
		return output
	@staticmethod
	def backward(ctx,gradOutput):
		input,=ctx.saved_tensors
		return gradOutput*2*input

# s=SquareFunction.apply
# input=torch.randn((5,5),requires_grad=True)
# print(input)
# output=s(input)
# print(output)
# gradients=torch.randn((5,5),requires_grad=True)
# print(gradients)
# #gradients=torch.tensor(gradients,dtype=torch.float64,requires_grad=True)
# output.backward(gradients)
# print(input.grad)

# # # gradcheck takes a tuple of tensors as input, check if your gradient
# # # evaluated with these tensors are close enough to numerical
# # # approximations and returns True if they all verify this condition.
# input = (Variable(torch.randn(5,5).double(), requires_grad=True),)
# test = gradcheck(SquareFunction.apply, input, eps=1e-6, atol=1e-4)
# print(test)