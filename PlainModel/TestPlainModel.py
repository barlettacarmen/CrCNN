import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import PlainModel2 as pm
import csv

#output of dnn,predicted class,true class
def saveAsCsv(result,predicted,labels):
	for res, p, l in zip(result,predicted,labels):
		res=str(res)
		p=str(p)
		l=str(l)
		res=res.split("\n")
		p=p.split("\n")
		l=l.split("\n")
		res=res[1:11]
		p=p[1:2]
		l=l[1:2]
		print(",".join(res+p+l))
	return

# MNIST Dataset
transform=transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.1307,), (0.3081,))
	])

testset = torchvision.datasets.MNIST(root='./MNISTdata', train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False,num_workers=2)


net=pm.plain_net(weights_path='PlainModelWoPad.pth')
net.eval()
image,label=testset[2]
output=net(image.unsqueeze_(0))
print(output)

'''
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
prediction = list()
correct=0
total=0
with torch.no_grad():
	for data in testloader:
		images, labels = data
		outputs = net(images)
		_, predicted = torch.max(outputs.data, 1)
		saveAsCsv(outputs,predicted,labels)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i].item()
			class_total[label] += 1


print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
for i in range(10):
	print('Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))'''

