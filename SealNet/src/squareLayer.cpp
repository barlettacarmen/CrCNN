#include "squareLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>
#include <thread>

using namespace std;
using namespace seal;


 SquareLayer::SquareLayer(string name,int th_count):
 	Layer(name),
 	th_count(th_count){

 	}

SquareLayer::SquareLayer(){}
SquareLayer::~SquareLayer(){}

//with threads
ciphertext3D SquareLayer::forward (ciphertext3D input){
	//Each thread computes the suare on images from z=from to z=to
	auto parallelForward=[&](ciphertext3D &input,ciphertext3D &result,int from, int to){


		for(int z=from;z<to;z++)
			for(int x=0;x<input[0].size();x++)
				for(int y=0;y<input[0][0].size();y++){
					/*assert(decryptor->invariant_noise_budget(input[z][x][y])>0);
					//decrypt
					decryptor->decrypt(input[z][x][y], tmp);
					dec=fraencoder->decode(tmp);
					//encrypt again
					encryptor->encrypt(fraencoder->encode(dec),input_bis);*/
					//square and relinerize
					evaluator->square(input[z][x][y],result[z][x][y],MemoryPoolHandle::Global());
					evaluator->relinearize(result[z][x][y],*ev_keys16);
				}

	};

	int z_size=input.size(),thread_z=0,from=0,to=0;
	ciphertext3D result(z_size, ciphertext2D(input[0].size(),vector<Ciphertext> (input[0][0].size())));
	vector<thread> th_vector;

	if(th_count>z_size)
		th_count=z_size;
	else if(th_count<=0)
			th_count=1;

	thread_z=z_size/th_count;
	
	
	for (int i = 0; i < th_count; i++){
		from=to;
    	if(i<th_count-1)
    		to+=thread_z;
    	//the last thread will compute also the remaning part
    	else
    		to+=thread_z + (z_size%th_count);

      	th_vector.emplace_back(parallelForward, ref(input),ref(result),from,to);

    }
    for (size_t i = 0; i < th_vector.size(); i++)
    {
        th_vector[i].join();
    }
    return result;


	
}

void SquareLayer::printLayerStructure(){
	//The real number of threads used can be determined after the 1st forward
	cerr<<"Square run with "<<th_count<<" threads"<<endl;
}

vector<ChooserPoly> SquareLayer::sqaureSimulator(vector<ChooserPoly> sim_input){

	for(int i=0;i<sim_input.size();i++){
		chooser_evaluator->square(sim_input[i]);
		//16=decomposition_bit_count
		chooser_evaluator->relinearize(sim_input[i],16);
	}

	return sim_input;

}