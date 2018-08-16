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
	int index=2;
	int zd=1,xd=28,yd=28;
	//Import MNIST dataset
    mnist::MNIST_dataset<vector, vector<float>, uint8_t> dataset =
    mnist::read_dataset<vector, vector, float, uint8_t>("../PlainModel/MNISTdata/raw");
    //Normalize dataset (by data'=(data-mean)/std)
     
        for(int i=0;i<28;i++){
        	for(int j=0;j<28;j++){
        cout<<dataset.test_images[index][i*28+j]<<"\t";
    	}
    cout<<endl;
    }  

    dataset.test_images=normalize(dataset.test_images,0.1307, 0.3081);

/*
    cout << "Nbr of training images = " << dataset.training_images.size() << endl;
    cout << "Nbr of training labels = " << dataset.training_labels.size() << endl;
    cout << "Nbr of test images = " << dataset.test_images.size() << endl;
    cout << "Nbr of test labels = " << dataset.test_labels.size() << endl; */

    for(int i=0;i<xd;i++){
        for(int j=0;j<yd;j++){
        cout<<dataset.test_images[index][i*xd+j]<<"\t";
    	}
    cout<<endl;
	} 
	
	cout<<(unsigned short)dataset.test_labels[index]<<endl;
	
	//Setting Encryption parameters
	setParameters();

	//Encrypting  1st image of testset

	ciphertext3D encrypted_image(zd,ciphertext2D(xd,vector<Ciphertext>(yd)));
	
	encrypted_image=encryptImage(dataset.test_images[index],zd,xd,yd);


	//Build Network structure reading model weights form file 
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildNetwork();
	net.printNetworkStructure();

	encrypted_image = net.forward(encrypted_image);




	//Decrypt Image
	floatCube image(1, vector<vector<float> > (10,vector<float>(1)));
	image=decryptImage(encrypted_image);

	//predicted image
	
	auto it = max_element(begin(image[0]), end(image[0]));
    cout << it - image[0].begin()<<endl;
	


 	//Deleting encryption parameters
	delParameters();









	return 0;

}