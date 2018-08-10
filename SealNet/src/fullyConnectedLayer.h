#ifndef FULLY_CONNECTED
#define FULLY_CONNECTED
#include <vector>
#include <string>
#include "seal/seal.h"
#include "layer.h"
#include "globals.h"

using namespace seal;

class FullyConnectedLayer: public Layer
{
public:
	string name;
	int in_dim, out_dim;
	//Potrebbero diventare vettori di plaintexts 
	vector<vector<float> > weights;
	vector<float> biases;
	FullyConnectedLayer(string name, int in_dim, int out_dim, vector<vector<float> >  & weights, vector<float> & biases );
	~FullyConnectedLayer();

	ciphertext3D forward (ciphertext3D input);
	//Given a ciphertext3D z,x,y, reshapes it with z=1, x= z*x*y , y=1. Row-by-row
	ciphertext3D reshapeInput(ciphertext3D input);
	Plaintext getWeight(int x_index,int y_index);
	Plaintext getBias(int x_index);
	void printLayerStructure();
	
	
};





#endif