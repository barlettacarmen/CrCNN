import h5py
import torch

model=torch.load('./PlainModel.pth')
#print(model)
h5f=h5py.File('./PlainModel.h5','w')
for key in model:
	h5f.create_dataset(key, data=model[key].numpy())
h5f.close()
