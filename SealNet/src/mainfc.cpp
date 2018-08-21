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
#include "fullyConnectedLayer.h"
#include "cnnBuilder.h"



using namespace std;
using namespace seal;

int main(){
	setParameters();
	CnnBuilder build("PlainModel.h5");
	int z_img=10,x_img=4,y_img=4,out=16;
	FullyConnectedLayer *fc_layer=build.buildFullyConnectedLayer("classifier.fc3",x_img*y_img*z_img,out,4);
	ciphertext3D image1(z_img,ciphertext2D(x_img,vector<Ciphertext>(y_img)));
	ciphertext3D image2(z_img,ciphertext2D(x_img,vector<Ciphertext>(y_img)));
	ciphertext3D result1(1,ciphertext2D(out,vector<Ciphertext>(1)));
	ciphertext3D result2(1,ciphertext2D(out,vector<Ciphertext>(1)));
	Plaintext tmp;
	float result_dec;

	for(int z=0;z<z_img;z++)
		for(int x=0;x<x_img;x++)
			for(int y=0;y<y_img;y++){
				encryptor->encrypt(intencoder->encode(x+y),image1[z][x][y]);
				encryptor->encrypt(intencoder->encode(x+y),image2[z][x][y]);
			}
	/*result1=fc_layer->forward(image1);
	cout<<"result w/o threads"<<endl;
	for(int x=0; x<out;x++){
		decryptor->decrypt(result1[0][x][0], tmp);
        result_dec= fraencoder->decode(tmp);
        cout<<result_dec<<"\t";
	}*/
	result2=fc_layer->forward(image2);
	cout<<"result with threads"<<endl;
	for(int x=0; x<out;x++){
		decryptor->decrypt(result2[0][x][0], tmp);
        result_dec= fraencoder->decode(tmp);
        cout<<result_dec<<"\t";
	}


	delParameters();

}