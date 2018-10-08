import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from CustomAvgPool2dModule import CustomAvgPooling
from collections import OrderedDict

class PlainTinyNet(nn.Module):
	def __init__(self):
		super(PlainTinyNet,self).__init__()
		#Input 28 x 28 x 1 ---> 24 x 24 x 32 ---> 12 x 12 x 32
		self.pool1_features=nn.Sequential(OrderedDict([
			('conv1', nn.Conv2d(1,32, kernel_size=5, stride=1)),
			('pool1', CustomAvgPooling(kernel_size=2,stride=2)),
			]))
		#12 x 12 x 32 ---> 8 x 8 x 64 ---> 4 x 4 x 64
		self.pool2_features=nn.Sequential(OrderedDict([
			('conv2', nn.Conv2d(32,64, kernel_size=5, stride=1)),
			('pool2', CustomAvgPooling(kernel_size=2,stride=2)),
			]))
		# 4 x 4 x 64 ---> 512 ---> 10
		self.classifier=nn.Sequential(OrderedDict([
			('fc3', nn.Linear(4*4*64,512)),
			('fc4', nn.Linear(512,10)),
			]))
	
	def pool1_forward(self,x):
		x = self.pool1_features(x)
		#print(x.shape,"after pool1")
		return x

	def pool2_forward(self,x):
		x=self.pool1_forward(x)
		x=self.pool2_features(x)
		#print(x.shape,"after pool2")
		return x
	
	def fc_forward(self,x):
		x=self.pool2_forward(x)
		x=x.view(-1, 4*4*64)
		#print(x.shape,"after view")
		x=self.classifier(x)
		return x
	
	def forward(self, x): 
		# x=self.pool1_features._modules['conv1'](x)
		# x=self.pool1_features._modules['pool1'](x)
		# x=self.pool1_features._modules['norm1'](x)
		# return x
		#return self.pool1_forward(x)
		return self.fc_forward(x)

# MNIST Dataset with features in range [-0.424,2.821]
transform=transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.1307,), (0.3081,))
	])

trainset=torchvision.datasets.MNIST(root='./MNISTdata',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True,num_workers=2)

#------------------------------
#Setting for Training
net=PlainTinyNet()
#Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Train
for epoch in range(2):

	running_loss = 0.0
	for i, data in enumerate(trainloader,0):
		# get the inputs
		inputs,labels=data
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		outputs=net(inputs)
		loss=criterion(outputs,labels)
		loss.backward()
		optimizer.step()
		#print statistics
		running_loss +=loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(),'./PlainModelTiny.pth')

