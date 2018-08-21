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
#include "squareLayer.h"



using namespace std;
using namespace seal;

int main(){
	setParameters();
	int z_img=2,x_img=6,y_img=6;
	float result_dec;
	Plaintext tmp;
	ciphertext3D image(z_img,ciphertext2D(x_img,vector<Ciphertext>(y_img)));

	SquareLayer sq_layer("square");

	for(int z=0;z<z_img;z++){
		for(int x=0;x<x_img;x++){
			for(int y=0;y<y_img;y++){
				encryptor->encrypt(fraencoder->encode(-0.4699977777777777777777777777),image[z][x][y]);
				//cout<<x+0.1<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
		cout<<endl;
	}

	image=sq_layer.forward(image);


	for(int z=0;z<z_img;z++){
		for(int x=0;x<x_img;x++){
			for(int y=0;y<y_img;y++){
				decryptor->decrypt(image[z][x][y],tmp);
				result_dec=fraencoder->decode(tmp);
				cout<<result_dec<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
		cout<<endl;
	}	


	delParameters();

}