#include <vector>
#include "seal/seal.h"
#include "globals.h"
#include "convolutionalLayer.h"

#include <ostream>

using namespace std;
using namespace seal;

int main(){
    setParameters();
    cout<<"Cazzo fai"<<endl;
    cout<<"Done"<<endl;
    ConvolutionalLayer * layer= new ConvolutionalLayer("prova",28,28,1,2,2,5,5,20);
    cout<<"done"<<endl<<flush;
    plaintext3D kernel(layer->xf, plaintext2D(layer->yf,vector<Plaintext>(layer->zd)));
    ciphertext2D convolved(layer->xo,vector<Ciphertext>(layer->yo));
    ciphertext3D image(layer->xd,ciphertext2D(layer->yd,vector<Ciphertext>(layer->zd)));
    //toy image with each data=5, encrypted
    IntegerEncoder intencoder(context->plain_modulus(),3);
    for(int i=0;i<layer->xd;i++)
        for(int j=0;j<layer->yd;j++)
            for(int z=0;z<layer->zd;z++){
                image[i][j].emplace_back(*parms);
                encryptor->encrypt(intencoder.encode(5),image[i][j][z]); 
                //cout << "encrypting for x:" << i << "y:" << j << "z:" << z <<endl << flush;           
            }
    cout<<"end of encryption"<<endl << flush;
    //Encoding fractional kernel in plaintext 
   
    FractionalEncoder fraencoder(context->plain_modulus(), context->poly_modulus(), 64, 32, 3);
    

    for(int i=0;i<layer->xf;i++)
        for(int j=0;j<layer->yf;j++)
            for(int z=0;z<layer->zd;z++){
            kernel[i][j][z]=fraencoder.encode(0.15);
            
    }
    cout<<"Done";

    convolved=layer->convolution3d(image,kernel);

    delete layer;
    delParameters();
}

/*int main(){

    setParameters();

    //Encoding fractional weights in plaintext 
   
    FractionalEncoder fraencoder(context->plain_modulus(), context->poly_modulus(), 64, 32, 3);
    

    const vector<float> weights { 
        0.15, 0.05, 0.05, 0.2, 0.05, 0.3, 0.1, 0.025, 0.075, 0.05 
    };
    vector<Plaintext> encoded_weights;
    for(int i=0;i<10;i++){
        encoded_weights.emplace_back(fraencoder.encode(weights[i]));
        cout << to_string(weights[i]).substr(0,6) << ((i < 9) ? ", " : "\n");
    }
    //Encrypting integer data
    IntegerEncoder intencoder(context->plain_modulus(),3);
    vector<uint64_t> data(10, 5);
    vector<Ciphertext> encrypted_data;
    for(int i=0;i<10;i++){
        //costruttore che alloca 2 ciphertext per ogni ciphertext, da modificare se non fai solo operazioni con plaintext
        encrypted_data.emplace_back(*parms);
        encryptor->encrypt(intencoder.encode(data[i]),encrypted_data[i]);
        cout << to_string(data[i]).substr(0,6) << ((i < 9) ? ", " : "\n");
    }
    //166 bits (per ogni ciphertext)
    cout << "Noise budget in fresh encryption: ";
     for(int i=0;i<10;i++)
        cout<< decryptor->invariant_noise_budget(encrypted_data[i]) << " bits";
    cout << "Size of a fresh encryption: " << encrypted_data[0].size() << endl;
   
   //Multiplication data*weights
    // vector<Ciphertext> res(10);
     for(int i=0;i<10;i++){
        evaluator->multiply_plain(encrypted_data[i],encoded_weights[i]);

     }
    

    cout<<"Done"<<endl;
    //164 bits 
    cout << "Noise budget of source after Multiplication: ";
        for(int i=0;i<10;i++)
        cout<< decryptor->invariant_noise_budget(encrypted_data[i]) << " bits";
    //cout << "Noise budget of destination after Multiplication: ";
    //for(int i=0;i<10;i++)
    //cout<< decryptor->invariant_noise_budget(res[i]) << " bits";



    vector<Plaintext> mul_result_plain(10);
    vector<float> mul_result(10);
    for(int i=0;i<10;i++){
    cout << "Decrypting result: ";
    decryptor->decrypt(encrypted_data[i], mul_result_plain[i]);
    mul_result[i] = fraencoder.decode(mul_result_plain[i]);
     cout << "Mult: " << mul_result[i]<< endl;
      }

    Ciphertext encrypted_result;
    cout << "Adding up all 10 ciphertexts: ";
    evaluator->add_many(encrypted_data, encrypted_result);
    cout << "Done" << endl;
    //163 bits
    cout << "Noise budget after addition: "
        << decryptor->invariant_noise_budget(encrypted_result) << " bits" << endl;

    Plaintext plain_result;
    cout << "Decrypting result: ";
    decryptor->decrypt(encrypted_result, plain_result);
    cout << "Done" << endl;
    //float or double
    float result = fraencoder.decode(plain_result);
    cout << "Weighted average: " << result<< endl;

    delParameters();
    


}*/



