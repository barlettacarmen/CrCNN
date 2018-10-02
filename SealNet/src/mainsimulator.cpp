#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <random>
#include <limits>
#include <fstream>

#include "seal/seal.h"
#include "globals.h"
#include "cnnBuilder.h"
#include <math.h>  


using namespace std;
using namespace seal;


int main(){

    cout << "Finding optimized parameters for network "<<endl;

    setChooserParameters();
    CnnBuilder build("PlainModelWoPad.h5");
    vector<ChooserPoly> sim_out=build.buildSimulatedNetwork(24,1);

    /*
    The optimal parameters are now computed using the select_parameters 
    function in ChooserEvaluator. It is possible to give this function the 
    results of several distinct computations (as ChooserPoly objects), all 
    of which are supposed to be possible to perform with the resulting set
    of parameters. However, here we have only one input ChooserPoly.
    */

    EncryptionParameters optimal_parms;
    chooser_evaluator->select_parameters(sim_out, 0, optimal_parms);
    cout << "Done" << endl;
    
    /*
    Create an SEALContext object for the returned parameters
    */
    SEALContext optimal_context(optimal_parms);
    print_parameters(optimal_parms);
    delChooserParameters();
    
    // /*
    // Do the parameters actually make any sense? We can try to perform the 
    // homomorphic computation using the given parameters and see what happens.
    // */
    // KeyGenerator keygen(optimal_context);
    // PublicKey public_key = keygen.public_key();
    // SecretKey secret_key = keygen.secret_key();
    // EvaluationKeys ev_keys;
    // keygen.generate_evaluation_keys(16, ev_keys);

    // Encryptor encryptor(optimal_context, public_key);
    // Evaluator evaluator(optimal_context);
    // Decryptor decryptor(optimal_context, secret_key);
    // IntegerEncoder encoder(optimal_context.plain_modulus(), 3);
    // FractionalEncoder fraencoder(optimal_context.plain_modulus(), optimal_context.poly_modulus(), 64, 32, 3);



}