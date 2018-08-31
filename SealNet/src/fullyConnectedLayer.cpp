#include "fullyConnectedLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>
#include <thread>
#include <ostream>
#include <fstream>

using namespace std;
using namespace seal;


FullyConnectedLayer::FullyConnectedLayer(string name, int in_dim, int out_dim,int th_count, plaintext2D  & weights, vector<Plaintext> & biases):
	Layer(name),
	in_dim(in_dim),out_dim(out_dim),
	th_count(th_count),
	weights(weights), biases(biases){

	if(th_count>out_dim)
		th_count=out_dim;	
	else if(th_count<=0)
	 		th_count=1;

	}
FullyConnectedLayer::FullyConnectedLayer(string name, int in_dim, int out_dim,int th_count, istream * infile):
	Layer(name),
	in_dim(in_dim),out_dim(out_dim),
	th_count(th_count){
		loadPlaintextParameters(infile);
		if(th_count>out_dim)
			th_count=out_dim;	
	else if(th_count<=0)
	 		th_count=1;
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
   return weights[x_index][y_index]; 
 }

Plaintext FullyConnectedLayer::getBias(int x_index){
   return biases[x_index]; 
 }


//Forward implemented with threads
ciphertext3D FullyConnectedLayer::forward(ciphertext3D input){

	int from=0,to=0,thread_rows=0;
	ciphertext3D result(1,ciphertext2D(out_dim,vector<Ciphertext>(1)));
	vector<thread> th_vector;

	//Each thread will work on a portion of computation (ax+b) making matrix product of rows from index "form" to intex "to"
	auto parallelForward=[&](ciphertext3D &input,ciphertext3D &result,int from, int to){
		vector<Ciphertext> tmp(in_dim);

		for(int i=from; i<to; i++){
			for(int j=0;j<in_dim;j++){
				//weight=getWeight(i,j);
				evaluator->multiply_plain(input[0][j][0],weights[i][j],tmp[j],MemoryPoolHandle::Global());
			}
		evaluator->add_plain(tmp[0],biases[i]);
		evaluator->add_many(tmp,result[0][i][0]);
		}

	};
	input=reshapeInput(input);


	thread_rows=out_dim/th_count;
	
	
	for (int i = 0; i < th_count; i++){
		from=to;
    	if(i<th_count-1)
    		to+=thread_rows;
    	else
    		to+=thread_rows + (out_dim%th_count);

      	th_vector.emplace_back(parallelForward, ref(input),ref(result),from,to);

    }
    for (size_t i = 0; i < th_vector.size(); i++)
    {
        th_vector[i].join();
    }
    return result;

}
//Forward implemented without threads
/*
ciphertext3D FullyConnectedLayer::forward(ciphertext3D input){
	cout<<"Begin forward"<<endl<<flush;
	Plaintext weight;
	ciphertext3D result(1,ciphertext2D(out_dim,vector<Ciphertext>(1)));
	vector<Ciphertext> tmp(in_dim);

	input=reshapeInput(input);

	//cout<<decryptor->invariant_noise_budget(input[0][6][0])<<endl;
	//cout<<input[0].size()<<endl;
	for(int i=0; i<out_dim; i++){
		for(int j=0;j<in_dim;j++){
			weight=getWeight(i,j);
			//cout<<decryptor->invariant_noise_budget(input[0][j][0])<<endl;
			assert(decryptor->invariant_noise_budget(input[0][j][0])>0);
			evaluator->multiply_plain(input[0][j][0],weight,tmp[j]);
			//cout<<decryptor->invariant_noise_budget(tmp[j])<<endl;
			//cout<<"Multiply"<<i<<endl<<flush;
		}
	//adding bias
	evaluator->add_plain(tmp[0],getBias(i));
	evaluator->add_many(tmp,result[0][i][0]);
	//cout<<decryptor->invariant_noise_budget(result[0][i][0])<<endl;
	}
	return result;
}*/

void FullyConnectedLayer::savePlaintextParameters(ostream * outfile){
		int i,j;
		for(i=0;i<out_dim;i++){
			for(j=0;j<in_dim;j++){
				weights[i][j].save(*outfile);
				outfile->flush();
			}
			biases[i].save(*outfile);
			outfile->flush();
		}
}
void FullyConnectedLayer::loadPlaintextParameters(istream * infile){		
		int i,j;
		vector<Plaintext> encoded_biases(out_dim);
		plaintext2D encoded_weights(out_dim,vector<Plaintext> (in_dim));

		for(i=0;i<out_dim;i++){
			for(j=0;j<in_dim;j++){
				encoded_weights[i][j].load(*infile);
				//weights[i][j].load(infile);
				}
			encoded_biases[i].load(*infile);
			//biases[i].load(infile);
		}
		weights=encoded_weights;
		biases=encoded_biases;
}

void FullyConnectedLayer::printLayerStructure(){
    cerr<<"Fully connected "<<name<<" : ("<<in_dim<<" -> "<<out_dim<<")"<<"run with "<<th_count<<" threads"<<endl;

}

FullyConnectedLayer::~FullyConnectedLayer(){}