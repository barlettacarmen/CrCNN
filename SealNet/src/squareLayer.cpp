#include "squareLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>

using namespace std;
using namespace seal;


 SquareLayer::SquareLayer(string name):
 	Layer(name){

 	}

SquareLayer::SquareLayer(){}
SquareLayer::~SquareLayer(){}

ciphertext3D SquareLayer::forward (ciphertext3D input){
	int x_size=input[0].size(), y_size=input[0][0].size(), z_size=input.size();
	int z,x,y;

	cout << "Size of a fresh encryption: " << input[0][0][0].size() << endl;
    cout << "Noise budget in fresh encryption: "
        << decryptor->invariant_noise_budget(input[0][0][0]) << " bits" << endl;

	for(z=0;z<z_size;z++)
		for(x=0;x<x_size;x++)
			for(y=0;y<y_size;y++){
				assert(decryptor->invariant_noise_budget(input[z][x][y])>0);
				evaluator->square(input[z][x][y]);

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
}

void SquareLayer::printLayerStructure(){
	cout<<"Square"<<endl;
}
