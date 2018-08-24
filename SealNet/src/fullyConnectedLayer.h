#ifndef FULLY_CONNECTED
#define FULLY_CONNECTED
#include <vector>
#include <string>
#include "seal/seal.h"
#include "layer.h"
#include "globals.h"
#include <ostream>
#include <fstream>

using namespace seal;

class FullyConnectedLayer: public Layer
{
public:
	string name;
	//th_count = # of threads to use to split the workload
	int in_dim, out_dim, th_count;
	plaintext2D weights;
	vector<Plaintext> biases;
	FullyConnectedLayer(string name, int in_dim, int out_dim,int th_count, plaintext2D  & weights, vector<Plaintext> & biases );
	//Weights and biases are loaded from file_name
	FullyConnectedLayer(string name, int in_dim, int out_dim,int th_count, istream * infile);
	~FullyConnectedLayer();

	ciphertext3D forward (ciphertext3D input);
	//Given a ciphertext3D z,x,y, reshapes it with z=1, x= z*x*y , y=1. Row-by-row
	ciphertext3D reshapeInput(ciphertext3D input);
	Plaintext getWeight(int x_index,int y_index);
	Plaintext getBias(int x_index);
	void savePlaintextParameters(ostream * outfile);
	void loadPlaintextParameters(istream * infile);

	void printLayerStructure();
	//
	//ciphertext3D forward_mod(ciphertext3D input);

	
	
};





#endif