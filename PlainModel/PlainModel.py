import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from CustomAvgPool2dModule import CustomAvgPooling
from SquareModule import Square
from collections import OrderedDict
import pickle as pkl

class PlainNet(nn.Module):
	def __init__(self):
		super(PlainNet,self).__init__()
		#Input 28 x 28 x 1 --> 12 x 12 x 20 --> 11 x 11 x 20
		self.pool1_features = nn.Sequential(OrderedDict([
			#('conv1', nn.Conv2d(1,20, kernel_size=5, stride=2, padding=2)),
			('conv1', nn.Conv2d(1,20, kernel_size=5, stride=2)),
			#('pool1', nn.AvgPool2d(kernel_size=2,stride=1)),
			('pool1', CustomAvgPooling()),
			#('norm1', nn.BatchNorm2d(20, affine=False)),
			]))
		# 11 x 11 x 20 --> 5 x 5 x 50 --> 4 x 4 x 50
		self.pool2_features = nn.Sequential(OrderedDict([
			('conv2', nn.Conv2d(20,50, kernel_size=3, stride=2)),
			#potresti  doverla spostare dopo norm2
			#('acti2', nn.ReLU(inplace=True)),
			('act1',Square()),
			('pool2', CustomAvgPooling()),
			#('pool2', nn.AvgPool2d(kernel_size=2,stride=1)),
			#('norm2', nn.BatchNorm2d(50, affine=False)),
			]))
		# 4 x 4 x 50 --> 500 --> 10
		self.classifier = nn.Sequential(OrderedDict([
			('fc3', nn.Linear(4*4*50, 500)),
			('fc4', nn.Linear(500,10)),
			]))
		
		self.functions_dict={
			'pool1': self.pool1_forward,
			'pool2': self.pool2_forward,
			'fc': self.fc_forward
		}

	def pool1_forward(self, x):
		#print(x.shape,"before pool1")
		x = self.pool1_features(x)
		#print(x.shape,"after pool1")
		return x

	def pool2_forward(self, x):
		x = self.pool1_forward(x)
		#print(x.shape,"before pool2")
		x = self.pool2_features(x)
		#print(x.shape,"after pool2")
		return x

	def fc_forward(self, x):
		x = self.pool2_forward(x)
		#print(x.shape,"before x.view")
		x = x.view(-1, 4 * 4 * 50)
		#print(x.shape,"after x.view")
		x = self.classifier(x)
		#print(x.shape,"after classifier")
		return x

	def forward(self, x): 
		# x=self.pool1_features._modules['conv1'](x)
		# x=self.pool1_features._modules['pool1'](x)
		# x=self.pool1_features._modules['norm1'](x)
		# return x
		#return self.pool1_forward(x)
		return self.fc_forward(x)

# MNIST Dataset with features in range [0,255]
transform=transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.1307,), (0.3081,))
	#transforms.Lambda(lambda x: x*255)
	])

trainset=torchvision.datasets.MNIST(root='./MNISTdata',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=2)
#------------------------------
#Setting for Training
net=PlainNet()
#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Train
for epoch in range(2):

	running_loss = 0.0
	for i, data in enumerate(trainloader,0):
		# get the inputs
		inputs,labels=data
		#inputs=torch.tensor(inputs,requires_grad=True)
		#labels=torch.tensor(labels,requires_grad=True)
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs=net(inputs)
		#outputs=torch.tensor(outputs,requires_grad=True)
		loss=criterion(outputs,labels)
		loss.backward()
		
		# print(grads['a'])
		# print(grads['b'])
		# print(grads['c'])
		# print(grads['d'])
		# print(grads['e'])
		# print(grads['f'])
		# print(grads['g'])
		# print(grads['h'])
		# print(grads['i'])
		# print(grads['j'])
		# print(grads['k'])
		# print(grads['l'])

		optimizer.step()
		#print statistics
		running_loss +=loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(),'./PlainModelWoBatchNorm.pth')


def plain_net(weights_path=None):
	model = PlainNet()
	# original saved file with DataParallel
	state_dict = torch.load(weights_path)
	# create new OrderedDict that does not contain `module.`
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		if(k != "pool1_features.norm1.num_batches_tracked" and k!="pool2_features.norm2.num_batches_tracked"):
			new_state_dict[k] = v
	# load params
	model.load_state_dict(new_state_dict)
	return model





