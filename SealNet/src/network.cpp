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
		/*Necessary when there is the need to find optimal plain_modulus*/
		/*If noise budget after the computation of the forward on a layer, is zero, than we reencrypt the output of the previous layer
		and repeat the forward for that layer*/
		int max_num_of_reencryptions=1;
		ciphertext3D output;

		for(int i=0; i<layers.size();i++){
			cerr<<i<<" "<<decryptor->invariant_noise_budget(input[0][0][0])<<endl;
			output=layers[i]->forward(input);
			if(decryptor->invariant_noise_budget(output[0][0][0])<=5){
				if(max_num_of_reencryptions>0){
					floatCube image=decryptImage(input);
					input=encryptImage(image);
					i--;
					max_num_of_reencryptions--;
					cout<<"Reencrypt due to out of noise budget. Still remaining "<<max_num_of_reencryptions<<endl;
					continue;
				}
				/* Used for optimalParametrsChooser: If max num of reencryptions has been reached throw exception telling which is the maximum layer that can be computed correclty*/
				else{
				 	throw OutOfBudgetException(i-1);
				}
			}
			input=deepCopyImage(output);
		}
		return output;
	}

	Network::~Network(){}

