#ifndef OPTIMAL_PARAMETERS_CHOOSER

#include <string>
#include "cnnBuilder.h"
using namespace std;
/* Returns the optimal plain_modulus for the network (which plain model is given in path_to_model), by performing a binary search between two values min and max,
keeping fixed th poly_modulus and the coeff_modulus to a big number, s.t the noise budget is big enough to perform all the computations.
If no plain_modulus is found, it is due to the fact that there are too much computations, so there will be the need to encrypt and reencrypt again the 
ciphertext during the computations.
Precondition: - a network trainded and saved in a txt file must be given;
			 - the corresponfing network structure must be defined in function buildNetwork of cnnBuilder
N.B. To optimize the search, the min_plain_modulus should be greater than the maximum/minimum data in input to the network
*/
uint64_t plainModulusBinarySearch(uint64_t min_plain_modulus=1<<2, uint64_t max_plain_modulus=1UL<<59, string path_to_model="");






#define OPTIMAL_PARAMETERS_CHOOSER
#endif