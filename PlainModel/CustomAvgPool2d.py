import torch
import math
from torch.autograd import gradcheck,Variable

class CustomAvgPoolingFunction(torch.autograd.Function):
	def __init__(self):
		super(CustomAvgPooling, self).__init__()

	@staticmethod
	def forward(ctx,input):
		kH=2
		kW=2
		dH=1
		dW=1
		ctx.save_for_backward(input)
		if(len(input.shape)==4):
			nbatch=input.shape[0]
		else: nbatch=1

		(nInputPlane,inputHeight,inputWidth,outputHeight,outputWidth)=shapeCheck(input)

		output=torch.zeros((nbatch,nInputPlane,outputHeight,outputWidth),requires_grad=True)
		for k in range(0,nInputPlane):
			for p in range(0,nbatch):
				for yy in range(0,outputHeight):
					for xx in range(0,outputWidth):
						hstart = yy * dH
						wstart = xx * dW
						hend = min(hstart + kH, inputHeight)
						wend = min(wstart + kW, inputWidth)
						hstart = max(hstart, 0)
						wstart = max(wstart, 0)
						sum = 0
						for ky in range (hstart,hend):
							for kx in range(wstart,wend):
								sum += input[p][k][ky][kx]
						output[p][k][yy][xx]=sum
						#print('Finished kernel')
		return output

	@staticmethod
	def backward(ctx,gradOutput):
		kH=2
		kW=2
		dH=1
		dW=1
		input, = ctx.saved_tensors
		if(len(input.shape)==4):
			nbatch=input.shape[0]
		else: nbatch=1
		gradInput=torch.zeros(input.shape,requires_grad=True)
		(nInputPlane,inputHeight,inputWidth,outputHeight,outputWidth)=shapeCheck(input)
		for k in range(0,nInputPlane):
			for p in range(0,nbatch):
				for yy in range(0,outputHeight):
					for xx in range(0,outputWidth):
						hstart = yy * dH
						wstart = xx * dW
						hend = min(hstart + kH, inputHeight)
						wend =	min(wstart + kW, inputWidth)
						hstart = max(hstart, 0)
						wstart = max(wstart, 0)

						for ky in range (hstart,hend):
							for kx in range(wstart,wend):
								gradInput[p][k][ky][kx]=gradOutput[p][k][yy][xx]
		return gradInput

def shapeCheck(input):
	kH=2
	kW=2
	dH=1
	dW=1
	ndim=len(input.shape)
	dimf=0
	dimh=1
	dimw=2
	if(ndim==4):
		dimf+=1
		dimh+=1
		dimw+=1
	nInputPlane=input.shape[dimh-1]
	inputHeight=input.shape[dimh]
	inputWidth=input.shape[dimw]

	outputHeight = (int)(math.floor((float)(inputHeight - kH) / dH)) + 1
	outputWidth  = (int)(math.floor((float)(inputWidth  - kW) / dW)) + 1
	#ensure that the last pooling starts inside the image
	#needed to avoid problems in ceil mode
	if ((outputHeight - 1)*dH >= inputHeight):
		outputHeight-=1
	if ((outputWidth  - 1)*dW >= inputWidth):
		outputWidth-=1
	if (outputWidth < 1 or outputHeight < 1):
		print("Output size is to small")
	return (nInputPlane,inputHeight,inputWidth,outputHeight,outputWidth)

# avg=CustomAvgPooling.apply
# input=torch.randn((4,6,10,10),requires_grad=True)
# print(input)
# output=avg(input)
# print(output)
# gradients=torch.randn((4,6,5,5),requires_grad=True)
# print(gradients)
# #gradients=torch.tensor(gradients,dtype=torch.float64,requires_grad=True)
# output.backward(gradients)
# print(input.grad)

# # gradcheck takes a tuple of tensors as input, check if your gradient
# # evaluated with these tensors are close enough to numerical
# # approximations and returns True if they all verify this condition.
# input = (Variable(torch.randn(1,1,20,20).double(), requires_grad=True),)
# test = gradcheck(CustomAvgPooling.apply, input, eps=1e-6, atol=1e-4)
# print(test)

