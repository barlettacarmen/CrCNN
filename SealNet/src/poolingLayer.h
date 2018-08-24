#ifndef POOLING_LAYER
#define POOLING_LAYER

#include "seal/seal.h"
#include "globals.h"
#include "layer.h"


using namespace seal;

class PoolingLayer: public Layer
{ /* This class implements a 2d "sum" Pooling on a xd by yd image with a filter("of ones") of size xf by yf, given xs and ys strides. */
public:
	int xd,yd,xs,ys,xf,yf,xo,yo,zo;
	PoolingLayer(string name,int xd,int yd, int zd, int xs, int ys,int xf, int yf);
	~PoolingLayer();

	ciphertext3D forward (ciphertext3D input);
	void printLayerStructure();
	//No parameters, so no need to implement these methods
	void savePlaintextParameters(ostream * outfile){};
	void loadPlaintextParameters(istream * infile){};
	
};





#endif