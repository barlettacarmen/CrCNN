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

//about 1.56 min needed (da rimisuarare)
Network setParametersAndSaveNetwork(int poly_modulus, uint64_t plain_modulus){
	setAndSaveParameters("pub_key.txt","sec_key.txt","eval_key.txt",poly_modulus,plain_modulus);
	//Build Network structure reading model weights form file 
	cout<<"keys done"<<endl<<flush;
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildAndSaveNetwork("encoded_model.txt");
	cout<<"bulit and saved net"<<endl<<flush;
	//net.printNetworkStructure();
	return net;

}
// about 0.935s
Network getParametersAndReadNetwork(int poly_modulus, uint64_t plain_modulus){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt",poly_modulus, plain_modulus);
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildNetwork("encoded_model.txt");
	//net.printNetworkStructure();
	return net;
}
//about 1m47.044s
Network getParametersAndSaveNetwork(int poly_modulus, uint64_t plain_modulus){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt",poly_modulus, plain_modulus);
	CnnBuilder build("PlainModelWoPad.h5");
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

/*int main(){

	vector<unsigned char>labels=loadMNISTestLabels("../PlainModel/MNISTdata/raw");
	floatCube image(1, vector<vector<float> > (10,vector<float>(1)));
	
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_forward(0);

	 Network net=getParametersAndReadNetwork();

	for(int i=0;i<2;i++){
		ciphertext3D encrypted_image=loadEncryptedImage(1, 28, 28, "./Cipher_Imgs/cipher_image_"+to_string(i+1)+".txt");

		time_start = chrono::high_resolution_clock::now();
		encrypted_image = net.forward(encrypted_image);
		time_end = chrono::high_resolution_clock::now();

		time_forward += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

		cout<<chrono::duration_cast<chrono::microseconds>(time_end - time_start).count()<<endl;

		//Decrypt Image
		
		image=decryptImage(encrypted_image);

		//predicted image
		
		auto it = max_element(begin(image[0]), end(image[0]));
	    cout << it - image[0].begin();

	    //true class
	    cout<<','<<(unsigned short)labels[i]<<endl;
	}

    cout<<"avg_forward_time= "<<time_forward.count()/2<<endl;
	
	delParameters();

}*/

int main(){

	/*Load and normalize your dataset*/
	vector<vector<float> > test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
	/*Load labels*/
	vector<unsigned char> predicted_labels=loadMNISTPlainModelPredictions("../PlainModel/predictionsPlainModelWoPad.csv");
	// chrono::high_resolution_clock::time_point time_start, time_end;
	// chrono::microseconds time_encrypt(0);
	// chrono::microseconds time_decrypt(0);

	int num_images_to_test=9999;
	int poly_modulus=4096;
	uint64_t plain_modulus=1UL<<29;
	
	cout<<"TEST IMAGES FROM 100 TO 9999 WITH n=4096, t=2^29, threads=40"<<endl;
	//Network net=setParametersAndSaveNetwork(poly_modulus,plain_modulus);
	Network net=getParametersAndReadNetwork(poly_modulus,plain_modulus);
	//cout<<"INDEX_IMG,T_LAYER_0,T_REENC_0,...,T_LAYER_N,T_REENC_N,T_ENC,T_DEC,PREDICTION"<<endl;
	cout<<"INDEX_IMG,PREDICTION"<<endl;

	/*Encrypt and test "num_images_to_test"*/
	for(int i=100;i<num_images_to_test;i++){
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
