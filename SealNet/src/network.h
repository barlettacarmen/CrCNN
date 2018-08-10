#ifndef NETWORK
#define NETWORK

#include "globals.h"
#include "layer.h"
#include <vector>

using namespace std;

class Network
{
public:

	vector<shared_ptr<Layer>> layers;

	Network();
	~Network();

	int getNumLayers(){return layers.size();}
	virtual shared_ptr<Layer> getLayer(int i){ return layers[i];}
	vector<shared_ptr<Layer> > & getLayers(){return layers;}
	void printNetworkStructure();
	ciphertext3D forward (ciphertext3D input);
	
};



#endif