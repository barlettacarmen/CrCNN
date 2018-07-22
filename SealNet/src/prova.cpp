#include <vector>
#include "seal/seal.h"
#include "globals.h"
#include "convolutionalLayer.h"

#include <ostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace seal;

int main(){
    //setParameters();
    initFromKeys("keys.txt");
    //dovresti runnare il test qualche volta per fare la media dei tempi
    chrono::high_resolution_clock::time_point time_start, time_end;
    //ostream
    /*ofstream outfile("ciphertexts.txt", ofstream::binary);
    ofstream outkeyfile("keys.txt", ofstream::binary);
    keygen->public_key().save(outkeyfile);
    keygen->secret_key().save(outkeyfile);*/
    //istream
    ifstream infile("ciphertexts.txt", ifstream::binary);
            /*
        These will hold the total times used by each operation.
        */
        chrono::microseconds time_encrypt_sum(0);
        chrono::microseconds time_decrypt_sum(0);
        chrono::microseconds time_convolve(0);

    ConvolutionalLayer * layer= new ConvolutionalLayer("prova",28,28,1,2,2,5,5,20);
    plaintext3D kernel(layer->zd, plaintext2D(layer->xf,vector<Plaintext>(layer->yf)));
    ciphertext3D convolved(layer->nf, ciphertext2D(layer->xo,vector<Ciphertext>(layer->yo)));
    ciphertext3D image(layer->zd,ciphertext2D(layer->xd,vector<Ciphertext>(layer->yd)));
    plaintext2D convolved_plain(layer->xo, vector<Plaintext>(layer->yo));
    vector<vector<float> > result(layer->xo, vector<float>(layer->yo));
    //toy image with each data=5, encrypted
    /*IntegerEncoder intencoder(context->plain_modulus(),3);

    time_start = chrono::high_resolution_clock::now();
    
    for(int i=0;i<layer->xd;i++)
        for(int j=0;j<layer->yd;j++)
            for(int z=0;z<layer->zd;z++){
                image[i][j].emplace_back(*parms);
                encryptor->encrypt(intencoder.encode(5),image[i][j][z]); 
                image[i][j][z].save(outfile);
                //cout << "encrypting for x:" << i << "y:" << j << "z:" << z <<endl << flush;           
            }
    time_end = chrono::high_resolution_clock::now();

    time_encrypt_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);


    cout<<"end of encryption"<<endl << flush;
    cout << "Noise budget in fresh encrypted pixel: ";
    cout<< decryptor->invariant_noise_budget(image[0][0][0]) << " bits"<<endl;
    */ 
    for(int z=0;z<layer->zd;z++)
        for(int i=0;i<layer->xd;i++)
            for(int j=0;j<layer->yd;j++){
                    image[z][i].emplace_back(*parms);
                    image[z][i][j].load(infile);
                }
    //Encoding fractional kernel in plaintext 
   
    FractionalEncoder fraencoder(context->plain_modulus(), context->poly_modulus(), 64, 32, 3);
    
for(int z=0;z<layer->zd;z++)
    for(int i=0;i<layer->xf;i++)
        for(int j=0;j<layer->yf;j++){
            kernel[z][i][j]=fraencoder.encode(0.15);
            
    }
    cout<<"Done Encoding Kernel"<<endl;
    //Do convolution

    time_start = chrono::high_resolution_clock::now();
     for(int z=0; z<layer->nf;z++)
        convolved[z]=layer->convolution3d(image,kernel);

    time_end = chrono::high_resolution_clock::now();

    time_convolve += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    cout << "Noise budget in a pixel of the convolved image: ";
    cout<< decryptor->invariant_noise_budget(convolved[0][0][0]) << " bits"<<endl<<flush;
    //Decrypting result

    time_start = chrono::high_resolution_clock::now();

    for(int z=0;z<layer->nf;z++)
        for(int i=0;i<layer->xo;i++){
            for(int j=0;j<layer->yo;j++){
                    decryptor->decrypt(convolved[z][i][j], convolved_plain[i][j]);
                    result[i][j]=fraencoder.decode(convolved_plain[i][j]);
                    cout<<result[i][j]<<" ";
        }
    cout<<""<<endl;
    }

    time_end = chrono::high_resolution_clock::now();

    time_decrypt_sum += chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    cout << "Encrypt: " << time_encrypt_sum.count()<< " microseconds" << endl;
    cout << "Decrypt: " << time_decrypt_sum.count() << " microseconds" << endl;
    cout<<"Convolution 1 kernel 2D: "<< time_convolve.count() << " microseconds" <<endl;
    cout.flush();

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



