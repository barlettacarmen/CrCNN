#ifndef CIPHERTEXT_IO
#define CIPHERTEXT_IO

#include "seal/seal.h"
#include <vector>

using namespace seal;
class CiphertextIO
{
public:
	vector<Ciphertext> data;
	int batch_size;
	int img_size;

	
	Ciphertext getData(int i){return data[i];}
	int getImgSize(){return img_size;}
	CiphertextIO(vector<Ciphertext> data, int batch_size,int img_size);
	~CiphertextIO(){}	
    
};




#endif