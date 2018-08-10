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
#include <typeinfo>

#include "mnist/mnist_reader.h"
#include "mnist/mnist_utils.h"
#include "seal/seal.h"
#include "globals.h"
#include "utils.h"
#include "cnnBuilder.h"
#include "H5Easy.h"



using namespace std;
using namespace seal;

int main(){
	//Import MNIST dataset
    mnist::MNIST_dataset<vector, vector<float>, uint8_t> dataset =
    mnist::read_dataset<vector, vector, float, uint8_t>("../PlainModel/MNISTdata/raw");
    //Normalize dataset (by data'=(data-mean)/std)
     /*
        for(int i=0;i<28;i++){
        	for(int j=0;j<28;j++){
        cout<<(float)dataset.test_images[0][i*28+j]<<"\t";
    	}
    cout<<endl;
    }    */

    dataset.test_images=normalize(dataset.test_images,0.1307, 0.3081);


   // mnist::normalize_dataset(dataset);

/*
    cout << "Nbr of training images = " << dataset.training_images.size() << endl;
    cout << "Nbr of training labels = " << dataset.training_labels.size() << endl;
    cout << "Nbr of test images = " << dataset.test_images.size() << endl;
    cout << "Nbr of test labels = " << dataset.test_labels.size() << endl; */

    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
        cout<<dataset.test_images[0][i*28+j]<<"\t";
    }
    cout<<endl;
} 
cout<<(unsigned short)dataset.test_labels[0]<<endl;
	//Build Network structure reading model weights form file 
	CnnBuilder build("PlainModelWoPad.h5");
	Network net=build.buildNetwork();
	net.printNetworkStructure();
	return 0;

}