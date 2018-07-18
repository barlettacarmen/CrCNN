#include "ciphertextIO.h"
#include "seal/seal.h"
#include <vector>

using namespace seal;


CiphertextIO::CiphertextIO(vector<Ciphertext> data, int batch_size,int img_size):
data(data), batch_size(batch_size),img_size(img_size){}

