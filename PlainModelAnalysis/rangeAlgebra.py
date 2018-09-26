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
range_min=(0,255)
range_max=(0,255)
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

def poolingRange(channel_ranges,window_size):
	return [mul(c,window_size) for c in channel_ranges]
	

channel_ranges=[(0,255)]
channel_ranges_out= convolutionalRange(channel_ranges, "pool1_features.conv1")
print("ranges", channel_ranges_out," range min ", range_min, "range_max " ,range_max)
print(len(channel_ranges_out))




