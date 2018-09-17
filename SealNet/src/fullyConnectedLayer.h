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
	bool weights_already_ntt;
	FullyConnectedLayer(string name, int in_dim, int out_dim,int th_count, plaintext2D  & weights, vector<Plaintext> & biases );
	//Weights and biases are loaded from file_name
	FullyConnectedLayer(string name, int in_dim, int out_dim,int th_count, istream * infile);
	~FullyConnectedLayer();

	ciphertext3D forward (ciphertext3D input);
	/*Given a ciphertext3D z,x,y, reshapes it with z=1, x= z*x*y , y=1. Row-by-row and transform the ciphertext in ntt
	If no rashping is needed, it only transforms the ciphertext to ntt*/
	ciphertext3D reshapeInput(ciphertext3D input);
	void transform_input_to_ntt(ciphertext3D &input);
	Plaintext getWeight(int x_index,int y_index);
	Plaintext getBias(int x_index);
	//void transform_weights_to_ntt();
	void savePlaintextParameters(ostream * outfile);
	void loadPlaintextParameters(istream * infile);

	void printLayerStructure();
	//
	//ciphertext3D forward_mod(ciphertext3D input);

	
	
};





#endif