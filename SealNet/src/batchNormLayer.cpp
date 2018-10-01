#include "batchNormLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>
#include <vector>
#include <ostream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace seal;

//The variance is already encoded as 1/sqrt(var+1e-05) in buildBatchNormLayer() in CnnBuilder class
BatchNormLayer::BatchNormLayer(string name, int num_channels,vector<Plaintext> & mean,vector<Plaintext> & var):
	Layer(name),
	num_channels(num_channels),
	mean(mean), var(var){
	}

BatchNormLayer::BatchNormLayer(string name, int num_channels,istream * infile):
	Layer(name),
	num_channels(num_channels)
	{
		loadPlaintextParameters(infile);
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

void BatchNormLayer::savePlaintextParameters(ostream * outfile){
	for(int i=0; i<num_channels;i++){
		mean[i].save(*outfile);
		var[i].save(*outfile);
		outfile->flush();
	}

}
void BatchNormLayer::loadPlaintextParameters(istream * infile){
	vector<Plaintext> encoded_mean(num_channels),encoded_var(num_channels);

		for(int i=0; i<num_channels;i++){
			encoded_mean[i].load(*infile);
			encoded_var[i].load(*infile);
		}

	mean=encoded_mean;
	var=encoded_var;

}

	Plaintext BatchNormLayer:: getMean(int index){
		return mean[index];
	}
	Plaintext BatchNormLayer:: getVar(int index){
		return var[index];
	}

void BatchNormLayer::printLayerStructure(){
	cerr<<"BatchNormLayer2D "<<name<<" :num_channels "<<num_channels<<endl;

}

vector<ChooserPoly> BatchNormLayer::batchNormSimulator(vector<ChooserPoly> sim_input, vector<float> & mean, vector<float> & var){

	int approx=1000;

	for(int i=0; i<sim_input.size();i++){
		chooser_evaluator->sub_plain(sim_input[i],chooser_encoder->encode(int(mean[i]*approx)));
		chooser_evaluator->multiply_plain(sim_input[i],chooser_encoder->encode(int(approx*1/sqrt(var[i] + 0.00001))));
	}

	return sim_input;

}