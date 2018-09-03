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

int main(){

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

}
