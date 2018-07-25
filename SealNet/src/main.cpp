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


#include "mnist/mnist_reader.h"
#include "seal/seal.h"
#include "globals.h"
#include "convolutionalLayer.h"
#include "cnnBuilder.h"


using namespace std;
using namespace seal;

int main()
{ //Import MNIST
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/Users/carmen/Desktop/UNI/Tesi/Tools/Pytorch/MNISTdata/raw");

    cout << "Nbr of training images = " << dataset.training_images.size() << endl;
    cout << "Nbr of training labels = " << dataset.training_labels.size() << endl;
    cout << "Nbr of test images = " << dataset.test_images.size() << endl;
    cout << "Nbr of test labels = " << dataset.test_labels.size() << endl;
    for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
        cout<<(unsigned short)dataset.test_images[0][i*28+j]<<"\t";
    }
    cout<<endl;
} cout<<(unsigned short)dataset.test_labels[0]<<endl;


    setParameters();

    CnnBuilder build("PlainModel.h5");
    ConvolutionalLayer layer= build.buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20);

    ciphertext3D image(layer.zd,ciphertext2D(layer.xd,vector<Ciphertext>(layer.yd)));
    
    //encrypting first image of test data
    for(int z=0;z<layer.zd;z++)
        for(int i=0;i<layer.xd;i++)
            for(int j=0;j<layer.yd;j++){
                image[z][i].emplace_back(*parms);
                encryptor->encrypt(intencoder->encode(dataset.test_images[0][i*layer.xd+j]),image[z][i][j]); 
                //image[i][j][z].save(outfile);
                //cout << "encrypting for x:" << i << "y:" << j << "z:" << z <<endl << flush;           
            }

}