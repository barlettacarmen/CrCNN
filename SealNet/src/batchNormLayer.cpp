#include "batchNormLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>
#include <vector>

using namespace std;
using namespace seal;

//The variance is already encoded as 1/sqrt(var+1e-05) in buildBatchNormLayer() in CnnBuilder class
BatchNormLayer::BatchNormLayer(string name, int num_channels,vector<Plaintext> mean,vector<Plaintext> var):
	Layer(name),
	num_channels(num_channels),
	mean(mean), var(var){
	}


BatchNormLayer::~BatchNormLayer(){}

ciphertext3D BatchNormLayer::forward (ciphertext3D input){
	int x_size=input[0].size(), y_size=input[0][0].size(), z_size=input.size();

	for(int z=0; z<z_size; ++z)
		for(int x=0;x<x_size; ++x)
			for(int y=0;y<y_size;++y){
				
				evaluator->sub_plain(input[z][x][y],mean[z]);
				evaluator->multiply_plain(input[z][x][y],var[z]);
			}
	return input;
}


void BatchNormLayer::printLayerStructure(){
	cout<<"BatchNormLayer2D "<<name<<" :num_channels "<<num_channels<<endl;

}