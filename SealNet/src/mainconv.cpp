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
#include "convolutionalLayer.h"
#include "cnnBuilder.h"



using namespace std;
using namespace seal;

int main(){
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	plaintext3D before,after;
	Plaintext before_b,after_b;
	ifstream infile;

	ConvolutionalLayer *conv1 = build.buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20,8,infile);
	ofstream outfile("conv_params.txt", ofstream::binary);
	conv1->savePlaintextParameters(outfile);
	outfile.close();
	infile.open("conv_params.txt", ifstream::binary);
	ConvolutionalLayer *conv2 = build.buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20,8,infile);
	infile.close();
	for(int n=0;n<20;n++){
		before=conv1->getKernel(n);
		after=conv2->getKernel(n);
		before_b=conv1->getBias(n);
		after_b=conv2->getBias(n);
        for(int z=0;z<1;z++){
            for(int i=0;i<5;i++){
                for(int j=0;j<5;j++){
                	cout<<"before weight"<<fraencoder->decode(before[z][i][j])<<endl<<flush;
                	cout<<"after weight"<<fraencoder->decode(after[z][i][j])<<endl<<flush;
                    if(before[z][i][j]!=after[z][i][j])
                    	cout<<"diversi pesi"<<endl<<flush;
                }
            }
        }
        cout<<"before bias"<<fraencoder->decode(before_b)<<endl<<flush;
        cout<<"after bias"<<fraencoder->decode(after_b)<<endl<<flush;
        if(before_b!=after_b)
        	cout<<"diversi bias"<<endl<<flush;
    }


	delParameters();

}