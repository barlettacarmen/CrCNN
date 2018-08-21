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
#include <algorithm>

#include "mnist/mnist_reader.h"
#include "seal/seal.h"
#include "globals.h"
#include "utils.h"
#include "cnnBuilder.h"
#include "H5Easy.h"



using namespace std;
using namespace seal;

int main(){
	
	int zd=1,xd=28,yd=28;
	//Import MNIST dataset
    mnist::MNIST_dataset<vector, vector<float>, uint8_t> dataset =
    mnist::read_dataset<vector, vector, float, uint8_t>("../PlainModel/MNISTdata/raw");
    
    //Normalize dataset (by data'=(data-mean)/std)
     
    dataset.test_images=normalize(dataset.test_images,0.1307, 0.3081);

	
	//Setting Encryption parameters
	setParameters();

	//Encrypting  1st image of testset

	ciphertext3D encrypted_image(zd,ciphertext2D(xd,vector<Ciphertext>(yd)));
	
	encrypted_image=encryptImage(dataset.test_images[index],zd,xd,yd);

	


 	//Deleting encryption parameters
	delParameters();









	return 0;

}