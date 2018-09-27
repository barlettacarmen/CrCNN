'''range_in=(min,Max)
if multiply for a negative weight, the range is inverted and scaled, otherwise it is only scaled'''
def mul(range_in,w):
	if(range_in[0]>range_in[1]):
		return "Error min is greater than max "+str(range_in)
	return (range_in[0] * w , range_in[1] * w) if(w>0) else (range_in[1] * w , range_in[0] * w)

'''with an addition/sub the range is only scaled'''
def add(range_in,w):
	if(range_in[0]>range_in[1]):
		return "Error min is greater than max "+str(range_in)
	return (range_in[0] + w , range_in[1] + w)

'''equivalent to calculate the codomain af a square function with a limited domain'''
def square(range_in):
	if(range_in[0]>range_in[1]):
		return "Error min is greater than max "+str(range_in)
	range_out=()
	if(range_in[0] <= 0 and range_in[1] > 0):
		range_out=(0 , max(range_in[0]**2 , range_in[1]**2 ))
	elif(range_in[0] >= 0 and range_in[1] >= 0):
		range_out=(range_in[0]**2 , range_in[1]**2)
	else:#if(range_in[0]<0 and range_in[1]<0)
		range_out=(range_in[1]**2 , range_in[0]**2)
	return range_out
	

import torch
import math
range_min=(-0.424,2.821)
range_max=(-0.424,2.821)
path_to_model="../PlainModel/PlainModelWoPad.pth"
model=torch.load(path_to_model)

def compareMinMaxglobal(range_in):
	global range_max
	global range_min
	if(range_in[0]< range_min[0]):
		range_min=range_in
	if(range_in[1]>range_max[1]):
		range_max=range_in
'''Input=list of ranges of len= number of channels in input
Output=list of ranges of len= number of channels in output
Computation= keep updated global min and max in all the single weights multiplication
Like a normal kernel but without the stride, and instead of the real images we have a simbolic pixel which value is between that ranges'''
def convolutionalRange(channel_ranges,layer_name):
	weights=model.get(layer_name+".weight")
	biases=model.get(layer_name+".bias")
	sizes=weights.size()
	channel_ranges_out=[(0,0) for i in range(sizes[0])]
	tmp_ranges=[(0,0) for i in range(sizes[1]*sizes[2]*sizes[3])]
	tmp_index=0
	for kernel in range(0,sizes[0]):
		for depth in range(0,sizes[1]):
			for x in range(0,sizes[2]):
				for y in range(0,sizes[3]):
					tmp_ranges[tmp_index]=mul(channel_ranges[depth],weights[kernel][depth][x][y])
					compareMinMaxglobal(tmp_ranges[tmp_index])
					tmp_index+=1

		channel_ranges_out[kernel]=tuple([sum(x) for x in zip(*tmp_ranges)])
		compareMinMaxglobal(channel_ranges_out[kernel])
		channel_ranges_out[kernel]=add(channel_ranges_out[kernel],biases[kernel])
		compareMinMaxglobal(channel_ranges_out[kernel])
		tmp_index=0
	return channel_ranges_out

'''Input=list of ranges of len= number of channels in input
Output=list of ranges of len= number of channels in output=channels in input
Computation= multiply each range for the size of the original pooling window.  
Keep updated global min and max after all single range multiplications.
Simulates a pooling summation, but all the pixels of the same channel are simbolyzed by a range of values, thus that range is only summed to itself'''
def poolingRange(channel_ranges,window_size):
	channel_ranges=[mul(c,window_size) for c in channel_ranges]
	[compareMinMaxglobal(c) for c in channel_ranges]
	return channel_ranges

'''Input=list of ranges of len= number of channels in input
Output=list of ranges of len= number of channels in output=channels in input
Computation= each range is batch normalized for its mean and var.  
Keep updated global min and max after all single range normalizations.
Simulates a BN summation, but all the pixels of the same channel are simbolyzed by a range of values.'''
def batchNormalizationRange(channel_ranges,layer_name):
	means=model.get(layer_name+".running_mean")
	variances=model.get(layer_name+".running_var")
	channel_ranges=[mul(add(c,-m),1/math.sqrt(v + 0.00001)) for c,m,v in zip(channel_ranges,means,variances)]
	[compareMinMaxglobal(c) for c in channel_ranges]
	return channel_ranges

'''Input=list of ranges of len= number of channels in input or = to #rows of prev fully connected
Output=list of ranges of len= to #rows of weights matrix
Computation= reshape if the dimension in input i smaller than the required one (eg 1st fully connected) by copying the same value of a channel
to obrain the dimension of an input image.
Keep updated global min and max after all single range computation.
Simulate a fully connected calculus.'''
def fullyConnectedRange(channel_ranges,layer_name):
	weights=model.get(layer_name+".weight")
	biases=model.get(layer_name+".bias")
	sizes=weights.size()
	tmp_ranges=[(0,0) for i in range(sizes[1])]
	channel_ranges_out=[(0,0) for i in range(sizes[0])]

	if(len(channel_ranges)<sizes[1]):
		num_copies=int(sizes[1]/len(channel_ranges))
		channel_ranges=[t for t in channel_ranges for i in range(num_copies)]

	for r in range(0,sizes[0]):
		for c in range(0,sizes[1]):
			tmp_ranges[c]=mul(channel_ranges[c],weights[r][c])
			compareMinMaxglobal(tmp_ranges[c])
		channel_ranges_out[r]=tuple([sum(x) for x in zip(*tmp_ranges)])
		compareMinMaxglobal(channel_ranges_out[r])
		channel_ranges_out[r]=add(channel_ranges_out[r],biases[r])
		compareMinMaxglobal(channel_ranges_out[r])
	return channel_ranges_out

'''Input=list of ranges of len= number of channels in input
Output=list of ranges of len= number of channels in output=channels in input
Computation= recompute ranges after square.
Keep updated global min and max after all square.
Simulates a square function on an image'''
def squareRange(channel_ranges):
	channel_ranges=[square(c) for c in channel_ranges]	
	[compareMinMaxglobal(c) for c in channel_ranges]
	return channel_ranges

channel_ranges=[(-0.424,2.821)]
channel_ranges= convolutionalRange(channel_ranges, "pool1_features.conv1")
print("After CONV1 ",len(channel_ranges))
#print("ranges CONV ", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges=poolingRange(channel_ranges,2*2)
print("After POOL1 ",len(channel_ranges))
#print("ranges POOL ", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges=batchNormalizationRange(channel_ranges,"pool1_features.norm1")
print("After BN1 ",len(channel_ranges))
#print("ranges BN ", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges= convolutionalRange(channel_ranges, "pool2_features.conv2")
print("After CONV2 ",len(channel_ranges))
#print("ranges CONV ", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges=squareRange(channel_ranges)
print("After SQUARE ", len(channel_ranges))
#print("ranges SQUARE", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges=poolingRange(channel_ranges,2*2)
print("After POOL2 ",len(channel_ranges))
#print("ranges POOL ", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges=batchNormalizationRange(channel_ranges,"pool2_features.norm2")
print("After BN2 ",len(channel_ranges))
#print("ranges BN ", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges=fullyConnectedRange(channel_ranges,"classifier.fc3")
print("After FC1 ",len(channel_ranges))
#print("ranges FC ", channel_ranges," range min ", range_min, "range_max " ,range_max)
channel_ranges=fullyConnectedRange(channel_ranges,"classifier.fc4")
print("After FC2 ",len(channel_ranges))
print("range min ", range_min, "range_max " ,range_max)








