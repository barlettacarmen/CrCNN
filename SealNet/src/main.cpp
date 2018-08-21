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
#include "poolingLayer.h"


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

    //CnnBuilder build("PlainModel.h5");
    //ConvolutionalLayer layer= build.buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20);
    PoolingLayer layer= PoolingLayer("pool1", 28 ,28, 2 , 2 , 2, 2);

    //1-->layer.zd
    ciphertext3D image(1,ciphertext2D(layer.xd,vector<Ciphertext>(layer.yd)));
    ciphertext3D convolved(layer.zo, ciphertext2D(layer.xo,vector<Ciphertext>(layer.yo)));
    Plaintext tmp;
    vector<vector<vector<float> > >  result(layer.zo,vector<vector<float> > (layer.xo,vector<float>(layer.yo)));
    
    //encrypting first image of test data 1-->layer.zd
    for(int z=0;z<1;z++)
        for(int i=0;i<layer.xd;i++)
            for(int j=0;j<layer.yd;j++){
                image[z][i].emplace_back(*parms);
                encryptor->encrypt(intencoder->encode(dataset.test_images[0][i*layer.xd+j]),image[z][i][j]); 
                //image[i][j][z].save(outfile);
                //cout << "encrypting for x:" << i << "y:" << j << "z:" << z <<endl << flush;           
            }
    cout << "Noise budget in fresh encrypted pixel: ";
    cout<< decryptor->invariant_noise_budget(image[0][0][0]) << " bits"<<endl;


    convolved=layer.forward(image);
    cout<<"xo: "<<layer.xo<<" yo: "<<layer.yo<<" zo:"<<layer.zo<<endl<<flush;

    cout << "Noise budget in a convolved pixel: ";
    cout<< decryptor->invariant_noise_budget(convolved[0][10][10]) << " bits"<<endl<<flush;

    for(int z=0;z<layer.zo;z++){
        for(int i=0;i<layer.xo;i++){
            for(int j=0;j<layer.yo;j++){
                    decryptor->decrypt(convolved[z][i][j], tmp);
                    result[z][i][j]= fraencoder->decode(tmp);
                    cout<<result[z][i][j]<<"\t";
        }
        cout<<endl;
    }
cout<<endl;
}

delParameters();

}