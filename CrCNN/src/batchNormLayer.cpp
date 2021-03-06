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

vector<ChooserPoly> BatchNormLayer::batchNormSimulator(vector<ChooserPoly> & sim_input, vector<float> & mean, vector<float> & var){
	cout<<"bn"<<flush;
	int approx=1000;

	for(int i=0; i<sim_input.size();i++){
		// int m=mean[i]*approx;
		// if(m==0)
		// 	sim_input[i]=chooser_evaluator->sub_plain(sim_input[i],31,1);
		// else
		// 	sim_input[i]=chooser_evaluator->sub_plain(sim_input[i],chooser_encoder->encode(m));
		sim_input[i]=chooser_evaluator->sub_plain(sim_input[i],encodeFractionalChooser(mean[i]));

		float v=1/sqrt(var[i] + 0.00001);
		// int v=approx/sqrt(var[i] + 0.00001);
		// if(v==0)
		// 	sim_input[i]=chooser_evaluator->multiply_plain(sim_input[i],31,1);
		// else
		// 	sim_input[i]=chooser_evaluator->multiply_plain(sim_input[i],chooser_encoder->encode(v));
		sim_input[i]=chooser_evaluator->multiply_plain(sim_input[i],encodeFractionalChooser(v));
	}

	cout<<" ended bn"<<flush;
	return sim_input;

}
ChooserPoly BatchNormLayer::batchNormSimulator(ChooserPoly sim_input){
	cout<<"bn"<<flush;


	sim_input=chooser_evaluator->sub_plain(sim_input,10,1);
	sim_input=chooser_evaluator->multiply_plain(sim_input,10,1);


	cout<<" ended bn"<<flush;
	return sim_input;

}