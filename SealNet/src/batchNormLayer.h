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
 
	BatchNormLayer(string name, int num_channels,vector<Plaintext> & mean,vector<Plaintext>  & var);
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

	/*Simulates batch norm computation during forward. The input is not a real chipertext image but it is a representation given by
	an upper bound on the number of non-zero coefficients in the Plaintext polynomial and the max_abs_value of these coefficients 
	(=1 if freshly encryption with base=3 or base=2)
	Input=The sim_input simulates an image and has just one pixel for each channel in input.
	Computation=  as in forward, but only one computation for each channel
	Output= one pixel for each channel in output (len=sim_input.size())*/
	static vector<ChooserPoly> batchNormSimulator(vector<ChooserPoly> sim_input, vector<float> & mean, vector<float> & var);

	
};
#endif