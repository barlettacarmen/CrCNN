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

    for(int k=0;k<11;k++){
        for(int i=0;i<28;i++){
        	for(int j=0;j<28;j++){
        cout<<dataset.test_images[k][i*28+j]<<"\t";
    	}
    cout<<endl;
    } 
	cout<<k<<","<<(unsigned short)dataset.test_labels[k]<<endl;
}

	return 0;

}