import h5py
import torch

model=torch.load('./PlainModelWoPad.pth')
#print(model)
h5f=h5py.File('./PlainModelWoPad.h5','w')
for key in model:
	h5f.create_dataset(key, data=model[key].numpy())
h5f.close()
