#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <random>
#include <limits>
#include <fstream>

#include "seal/seal.h"
#include "globals.h"
#include "network.h"
#include "cnnBuilder.h"
#include "utils.h"


using namespace std;
using namespace seal;

enum exit_status_forward{SUCCESS, OUT_OF_BUDGET,MISPREDICTED};
string model="PlainModelTiny.h5";

//about 1.56 min needed (da rimisuarare)
Network setParametersAndSaveNetwork(int poly_modulus, uint64_t plain_modulus){
	setAndSaveParameters("pub_key.txt","sec_key.txt","eval_key.txt",poly_modulus,plain_modulus);
	//Build Network structure reading model weights form file 
	cout<<"keys done"<<endl<<flush;
	CnnBuilder build(model);
	Network net=build.buildAndSaveNetwork("encoded_model.txt");
	cout<<"bulit and saved net"<<endl<<flush;
	//net.printNetworkStructure();
	return net;

}
// about 0.935s
Network getParametersAndReadNetwork(int poly_modulus, uint64_t plain_modulus){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt",poly_modulus, plain_modulus);
	CnnBuilder build(model);
	Network net=build.buildNetwork("encoded_model.txt");
	//net.printNetworkStructure();
	return net;
}
//about 1m47.044s
Network getParametersAndSaveNetwork(int poly_modulus, uint64_t plain_modulus){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt",poly_modulus, plain_modulus);
	CnnBuilder build(model);
	Network net=build.buildAndSaveNetwork("encoded_model.txt");
	//net.printNetworkStructure();
	return net;
}
//about 2 sec on server
void getParametersAndSaveImages(int from, int to,int poly_modulus, uint64_t plain_modulus){
	vector<vector<float> > test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt",poly_modulus, plain_modulus);
	for(int i=from; i<to;i++)
		encryptAndSaveImage(test_set[i],1,28,28,"./Cipher_Imgs/cipher_image_"+to_string(i+1)+".txt");
	
}


int main(){

	/*Load and normalize your dataset*/
	vector<vector<float> > test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
	/*Load labels*/
	vector<unsigned char> predicted_labels=loadMNISTPlainModelPredictions("../PlainModel/predictionsPlainModelTiny.csv");
	// chrono::high_resolution_clock::time_point time_start, time_end;
	// chrono::microseconds time_encrypt(0);
	// chrono::microseconds time_decrypt(0);

	int num_images_to_test=100;
	int poly_modulus=2048;
	uint64_t plain_modulus=1UL<<18;
	
	cout<<"TEST TINY TIME 2^18"<<endl;
	Network net=setParametersAndSaveNetwork(poly_modulus,plain_modulus);
	//Network net=getParametersAndReadNetwork(poly_modulus,plain_modulus);
	cout<<"INDEX_IMG,T_LAYER_0,...,T_REENC,T_LAYER_4,T_LAYER_5,PREDICTION"<<endl;
	//cout<<"INDEX_IMG,PREDICTION"<<endl;

	/*Encrypt and test "num_images_to_test"*/
	for(int i=0;i<num_images_to_test;i++){
		cout<<"OUTPUT: "<<i<<",";
		exit_status_forward ret_value=SUCCESS;
		//time_start = chrono::high_resolution_clock::now();
		ciphertext3D encrypted_image=encryptImage(test_set[i], 1, 28, 28);
		//time_end= chrono::high_resolution_clock::now();
		//time_encrypt= chrono::duration_cast<chrono::microseconds>(time_end - time_start);
		try{
			encrypted_image = net.forward(encrypted_image);
			}
		catch(OutOfBudgetException& e){
			/* If a reencryption has already been performed*/
			cout<<"Maximum layer computed is "<<e.last_layer_computed<<" exit due to OUT_OF_BUDGET"<<endl;
			ret_value=OUT_OF_BUDGET;
		}
		//time_start = chrono::high_resolution_clock::now();
		floatCube image=decryptImage(encrypted_image);
		//time_end= chrono::high_resolution_clock::now();
		//time_decrypt= chrono::duration_cast<chrono::microseconds>(time_end - time_start);
		auto it = max_element(begin(image[0]), end(image[0]));
		auto predicted = it - image[0].begin();
		/*If one of the tested images returns a wrong prediction, we need to find new plain_modulus*/
		if(predicted!=predicted_labels[i]){
			ret_value=MISPREDICTED;
			}
		//cout<<time_encrypt.count()<<","<<time_decrypt.count()<<",";
		cout<< (ret_value == SUCCESS ? "Success," : (ret_value == OUT_OF_BUDGET ? "Out of Budget," : "Mispredicted,") ) << endl;
	}

	delParameters();

}
