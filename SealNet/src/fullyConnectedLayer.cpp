#include "fullyConnectedLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>

using namespace std;
using namespace seal;


FullyConnectedLayer::FullyConnectedLayer(string name, int in_dim, int out_dim, vector<vector<float> >  & weights, vector<float> & biases):
	Layer(name),
	in_dim(in_dim),out_dim(out_dim),
	weights(weights), biases(biases){

	}
ciphertext3D FullyConnectedLayer::reshapeInput(ciphertext3D input){
	int x_size=input[0].size(), y_size=input[0][0].size(), z_size=input.size();
	//cout<<z_size<<" "<<x_size<<" "<<y_size<<endl;
	int z,x,y;
	if(z_size!=1 && y_size!=1){
		ciphertext3D reshaped_input(1,ciphertext2D(in_dim,vector<Ciphertext>(1)));
		for (int i = 0; i < in_dim; ++i)
		{
			z=i/(x_size*y_size);
			x=i/y_size - (x_size * z);
			y=i%y_size;
			//cout<<z<<" "<<x<<" "<<y<<endl;
			reshaped_input[0][i][0]= Ciphertext(input[z][x][y]);
		}
		return reshaped_input;
	}

	return input;
}

Plaintext FullyConnectedLayer::getWeight(int x_index,int y_index){
   return fraencoder->encode(weights[x_index][y_index]); 
 }

Plaintext FullyConnectedLayer::getBias(int x_index){
   return fraencoder->encode(biases[x_index]); 
 }

ciphertext3D FullyConnectedLayer::forward(ciphertext3D input){
	cout<<"Begin forward"<<endl<<flush;
	Plaintext weight;
	ciphertext3D result(1,ciphertext2D(out_dim,vector<Ciphertext>(1)));
	vector<Ciphertext> tmp(in_dim);

	input=reshapeInput(input);

	cout<<decryptor->invariant_noise_budget(input[0][6][0])<<endl;
	//cout<<input[0].size()<<endl;
	for(int i=0; i<out_dim; i++){
		for(int j=0;j<in_dim;j++){
			weight=getWeight(i,j);
			cout<<decryptor->invariant_noise_budget(input[0][j][0])<<endl;
			assert(decryptor->invariant_noise_budget(input[0][j][0])>0);
			evaluator->multiply_plain(input[0][j][0],weight,tmp[j]);
			cout<<decryptor->invariant_noise_budget(tmp[j])<<endl;
			cout<<"Multiply"<<i<<endl<<flush;
		}
	//adding bias
	evaluator->add_plain(tmp[0],getBias(i));
	evaluator->add_many(tmp,result[0][i][0]);
	cout<<decryptor->invariant_noise_budget(result[0][i][0])<<endl;
	}
	return result;
}

void FullyConnectedLayer::printLayerStructure(){
    cout<<"Fully connected "<<name<<" : ("<<in_dim<<" -> "<<out_dim<<")"<<endl;

}

FullyConnectedLayer::~FullyConnectedLayer(){}