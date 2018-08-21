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


#include "seal/seal.h"
#include "globals.h"
#include "network.h"
#include "cnnBuilder.h"
#include "utils.h"


using namespace std;
using namespace seal;
//about 1.56 min needed
Network setParametersAndNetworkFirstTime(){
	setAndSaveParameters("pub_key.txt","sec_key.txt","eval_key.txt");
	//Build Network structure reading model weights form file 
	cout<<"keys done"<<endl<<flush;
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildAndSaveNetwork("encoded_model.txt");
	net.printNetworkStructure();
	return net;

}
// about 0.935s 
Network getParametersAndNetworkSecondTime(){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildNetwork("encoded_model.txt");
	net.printNetworkStructure();
	return net;
}
//about 0m41.083s
void getParametersAndSaveImages(int from, int to){
	vector<vector<float> > test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	for(int i=from; i<to;i++)
		encryptAndSaveImage(test_set[i],1,28,28,"./Cipher_Imgs/cipher_image_"+to_string(i+1)+".txt");
	
}

int main(){
	Network net=getParametersAndNetworkSecondTime();
	//getParametersAndSaveImages(1,2);
	/*initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	//Network net=build.buildNetwork();
	Network net=build.buildAndSaveNetwork("encoded_model.txt");*/
	ciphertext3D encrypted_image=loadEncryptedImage(1, 28, 28, "./Cipher_Imgs/cipher_image_1.txt");
	encrypted_image = net.forward(encrypted_image);

	//Decrypt Image
	floatCube image(1, vector<vector<float> > (10,vector<float>(1)));
	image=decryptImage(encrypted_image);

	//predicted image
	
	auto it = max_element(begin(image[0]), end(image[0]));
    cout << it - image[0].begin()<<endl;
	
	delParameters();

}
