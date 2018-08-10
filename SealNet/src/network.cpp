#include "network.h"
#include "globals.h"
#include "seal/seal.h"
#include "layer.h"
using namespace std;
using namespace seal;


	Network::Network(){}
	
	void Network::printNetworkStructure(){
		for(int i=0; i<layers.size();i++){
			cout<<"("<<i<<") : ";
			layers[i]->printLayerStructure();
			cout<<endl;
		}
		

	}

	ciphertext3D Network::forward (ciphertext3D input){
		for(int i=0; i<layers.size();i++){
			input=layers[i]->forward(input);
		}
		return input;
	}

	Network::~Network(){}

