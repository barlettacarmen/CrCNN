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
//about 2 sec on server
void getParametersAndSaveImages(int from, int to){
	vector<vector<float> > test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	for(int i=from; i<to;i++)
		encryptAndSaveImage(test_set[i],1,28,28,"./Cipher_Imgs/cipher_image_"+to_string(i+1)+".txt");
	
}

int main(){



	// //getParametersAndSaveImages(1,2);
	/*initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildAndSaveNetwork("encoded_model.txt");
	
	//controlla cosa c'è in encoded_model.txt
	vector<Plaintext> p(1000);
	//initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	ifstream infile("encoded_model.txt", ifstream::binary);
		for(size_t i=520;i<1000;i++){
			p[i].load(infile);
			cout<<fraencoder->decode(p[i])<<endl<<flush;
		}
	infile.close();*/
/*
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildNetwork();*/

	Network net=getParametersAndNetworkSecondTime();
	/*plaintext3D before;
	Plaintext before_b;
	shared_ptr<Layer> base=net.getLayer(0);
	shared_ptr<ConvolutionalLayer> conv1 =
               dynamic_pointer_cast<ConvolutionalLayer> (base);

    cout<<"Convolutional1"<<endl;
    for(int n=0;n<20;n++){
		before=conv1->getKernel(n);
		before_b=conv1->getBias(n);
		cout<<n<<endl;
        for(int z=0;z<1;z++){
            for(int i=0;i<5;i++){
                for(int j=0;j<5;j++){
                	cout<<"before weight"<<fraencoder->decode(before[z][i][j])<<endl<<flush;
                }
            }
        }
        cout<<"before bias"<<fraencoder->decode(before_b)<<endl<<flush;
    }

    cout<<"BatchNorm1"<<endl;
    base=net.getLayer(2);
	shared_ptr<BatchNormLayer> bn1 =
               dynamic_pointer_cast<BatchNormLayer> (base);
    	for(int i=0;i<20;i++){
		before_b=bn1->getMean(i);
		cout<<"before mean"<<fraencoder->decode(before_b)<<endl<<flush;
		before_b=bn1->getVar(i);
		cout<<"before var"<<fraencoder->decode(before_b)<<endl<<flush;

	}*/


	ciphertext3D encrypted_image=loadEncryptedImage(1, 28, 28, "./Cipher_Imgs/cipher_image_2.txt");
	encrypted_image = net.forward(encrypted_image);

	//Decrypt Image
	floatCube image(1, vector<vector<float> > (10,vector<float>(1)));
	image=decryptImage(encrypted_image);

	//predicted image
	
	auto it = max_element(begin(image[0]), end(image[0]));
    cout << it - image[0].begin()<<endl;
	
	delParameters();

}
