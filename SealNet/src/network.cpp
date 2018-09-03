#include "network.h"
#include "globals.h"
#include "seal/seal.h"
#include "layer.h"
#include <cassert>
using namespace std;
using namespace seal;


	Network::Network(){}
	
	void Network::printNetworkStructure(){
		for(int i=0; i<layers.size();i++){
			cerr<<"("<<i<<") : ";
			layers[i]->printLayerStructure();
			cout<<endl;
		}
		

	}

	ciphertext3D Network::forward (ciphertext3D input){
		for(int i=0; i<layers.size();i++){
			cerr<<i<<" "<<decryptor->invariant_noise_budget(input[0][0][0])<<endl;
			assert(decryptor->invariant_noise_budget(input[0][0][0])>0);
			input=layers[i]->forward(input);
		}
		return input;
	}

	Network::~Network(){}

