#ifndef BATCH_NORM_LAYER
#define BATCH_NORM_LAYER

#include "seal/seal.h"
#include "globals.h"
#include "layer.h"
#include <string>
#include <vector>


class BatchNormLayer: public Layer
{
public:
	int num_channels;
	vector<Plaintext> mean;
	vector<Plaintext> var;
 
	BatchNormLayer(string name, int num_channels,vector<Plaintext> mean,vector<Plaintext> var);
	//Initilaize from file
	BatchNormLayer(string name, int num_channels,istream * infile);
	~BatchNormLayer();
	//For each pixel x--> x'=[x-mean(x)]*(1/sqrt(var(x)+1e-05))
	//The mean and standard-deviation are calculated per-dimension over the mini-batches see: https://pytorch.org/docs/stable/nn.html#batchnorm2d
	ciphertext3D forward (ciphertext3D input);
	Plaintext getMean(int index);
	Plaintext getVar(int index);
	void savePlaintextParameters(ostream * outfile);
	void loadPlaintextParameters(istream * infile);
	void printLayerStructure();

	
};
#endif