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
Network setParametersAndSaveNetwork(){
	setAndSaveParameters("pub_key.txt","sec_key.txt","eval_key.txt");
	//Build Network structure reading model weights form file 
	cout<<"keys done"<<endl<<flush;
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildAndSaveNetwork("encoded_model.txt");
	net.printNetworkStructure();
	return net;

}
// about 0.935s
Network getParametersAndReadNetwork(){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildNetwork("encoded_model.txt");
	//net.printNetworkStructure();
	return net;
}
//about 1m47.044s
Network getParametersAndSaveNetwork(){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildAndSaveNetwork("encoded_model.txt");
	//net.printNetworkStructure();
	return net;
}
//about 2 sec on server
void getParametersAndSaveImages(int from, int to){
	vector<vector<float> > test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
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
	vector<unsigned char> labels=loadMNISTestLabels("../PlainModel/MNISTdata/raw");

	int num_images_to_test=1;
	exit_status_forward ret_value=SUCCESS;
	cout<<"Testing "<<endl;
	
	setParameters(4096,1UL<<20);
	cout<<"keys done"<<endl<<flush;
	/*Encode the network  saved in path_to model*/
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildNetwork();
	cout<<"Built network"<<endl;
	/*Encrypt and test "num_images_to_test"*/
	for(int i=0;i<num_images_to_test;i++){
		cout<<"Encrypting image "<<i<<" ";
		ciphertext3D encrypted_image=encryptImage(test_set[i], 1, 28, 28);
		cout<<"Testing image "<<i<<endl;
		try{
			encrypted_image = net.forward(encrypted_image);
			}
		catch(OutOfBudgetException& e){
			/* If a reencryption has already been performed*/
			cout<<"Maximum layer computed is "<<e.last_layer_computed<<" exit due to OUT_OF_BUDGET"<<endl;
			ret_value=OUT_OF_BUDGET;
			break;
		}
		floatCube image=decryptImage(encrypted_image);

		auto it = max_element(begin(image[0]), end(image[0]));
		auto predicted = it - image[0].begin();
		/*If one of the tested images returns a wrong prediction, we need to find new plain_modulus*/
		if(predicted!=labels[i]){
			ret_value=MISPREDICTED;
			break;
			}
	}
	cout<< (ret_value == SUCCESS ? "Success" : (ret_value == OUT_OF_BUDGET ? "Out of Budget" : "Mispredicted") ) << endl;
	delParameters();
	//return ret_value;
}
