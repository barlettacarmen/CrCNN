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
		Plaintext tmp;
		float dec;
		Ciphertext input_bis;

		for(int z=from;z<to;z++)
			for(int x=0;x<input[0].size();x++)
				for(int y=0;y<input[0][0].size();y++){
					assert(decryptor->invariant_noise_budget(input[z][x][y])>0);
					//decrypt
					decryptor->decrypt(input[z][x][y], tmp);
					dec=fraencoder->decode(tmp);
					//encrypt again
					encryptor->encrypt(fraencoder->encode(dec),input_bis);
					//square and relinerize
					evaluator->square(input_bis,result[z][x][y],MemoryPoolHandle::Global());
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

/*
ciphertext3D SquareLayer::forward (ciphertext3D input){
	int x_size=input[0].size(), y_size=input[0][0].size(), z_size=input.size();
	int z,x,y;
	floatCube image(50, vector<vector<float> > (5,vector<float>(5)));
	Plaintext tmp;

	cout << "Size of input encryption: " << input[0][0][0].size() << endl;
    cout << "Noise budget in fresh encryption: "
        << decryptor->invariant_noise_budget(input[0][0][0]) << " bits" << endl;

    	//provo a decriptare e recriptare prima della square
	image=decryptImage(input);

	for(z=0;z<z_size;z++)
		for(x=0;x<x_size;x++)
			for(y=0;y<y_size;y++){
				assert(decryptor->invariant_noise_budget(input[z][x][y])>0);

				tmp=fraencoder->encode(image[z][x][y]);
				encryptor->encrypt(tmp,input[z][x][y]);
				////
				evaluator->square(input[z][x][y]);
				//evaluator->multiply(input[z][x][y],input[z][x][y]);

				cout << "Size after squaring: " << input[z][x][y].size() << endl;
				cout << "Noise budget after squaring: "
        		<< decryptor->invariant_noise_budget(input[z][x][y]) << " bits" << endl;

        		evaluator->relinearize(input[z][x][y],*ev_keys16);


				cout << "Size after relinearization: " << input[z][x][y].size() << endl;
				cout << "Noise budget after relinearizing (dbc = "
				    << ev_keys16->decomposition_bit_count() << "): "
				    << decryptor->invariant_noise_budget(input[z][x][y]) << " bits" << endl;
			}

	return input;
}*/

void SquareLayer::printLayerStructure(){
	//The real number of threads used can be determined after the 1st forward
	cout<<"Square run with "<<th_count<<" threads"<<endl;
}
