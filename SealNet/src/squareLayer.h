#ifndef SQUARE_LAYER
#define SQUARE_LAYER

#include "seal/seal.h"
#include "globals.h"
#include "layer.h"

class SquareLayer: public Layer
{ /*This class  implements an approximate activation function (the square function) by just calling SEAL square function*/
public:
	int th_count;
	SquareLayer(string name,int th_count);
	SquareLayer();
	~SquareLayer();

	ciphertext3D forward (ciphertext3D input);
	void printLayerStructure();
	//No parameters, so no need to implement these methods
	void savePlaintextParameters(ostream * outfile){};
	void loadPlaintextParameters(istream * infile){};
};


#endif