#include "globals.h"
#include "seal/seal.h"
#include <fstream>
#include <string>
#include <ostream>

using namespace seal;
using namespace std;

EncryptionParameters *parms;
SEALContext * context;
KeyGenerator * keygen;
Encryptor * encryptor;
Decryptor * decryptor;
Evaluator * evaluator;
IntegerEncoder * intencoder;
FractionalEncoder * fraencoder;
EvaluationKeys * ev_keys16;

void setParameters(){

  //4096,1<<20
	parms = new EncryptionParameters();
	parms->set_poly_modulus("1x^4096 + 1");
  parms->set_coeff_modulus(coeff_modulus_128(4096));
  //parms->set_coeff_modulus(*small_mods_60bits);
    //parms->set_poly_modulus("1x^32768 + 1");
    //parms->set_coeff_modulus(coeff_modulus_128(32768));

    //parms->set_plain_modulus(1099511922689);
    //parms->set_plain_modulus(4000000000);
  parms->set_plain_modulus(1<<20);


    context = new SEALContext(*parms);
    keygen= new KeyGenerator(*context);
    auto public_key = keygen->public_key();
    auto secret_key= keygen->secret_key();
    encryptor = new Encryptor(*context, public_key);
    decryptor = new Decryptor(*context,secret_key);
    evaluator = new Evaluator(*context);
    intencoder = new IntegerEncoder(context->plain_modulus(),3);
    //64,32
    fraencoder =  new FractionalEncoder(context->plain_modulus(), context->poly_modulus(), 64, 32, 3);
   	ev_keys16 = new EvaluationKeys();
    keygen->generate_evaluation_keys(16, *ev_keys16);
   
}

void setAndSaveParameters(string public_key_path,string secret_key_path,string evaluation_key_path){
    ofstream pubfile(public_key_path, ofstream::binary);
    ofstream secfile(secret_key_path, ofstream::binary);
    ofstream evalfile(evaluation_key_path, ofstream::binary);

    setParameters();

    keygen->public_key().save(pubfile);
    keygen->secret_key().save(secfile);
    ev_keys16->save(evalfile);
    
    pubfile.close();
    secfile.close();
    evalfile.close();


}


void initFromKeys(string public_key_path,string secret_key_path,string evaluation_key_path){
	ifstream pubfile(public_key_path, ifstream::binary);
  ifstream secfile(secret_key_path, ifstream::binary);
  ifstream evalfile(evaluation_key_path, ifstream::binary);

	parms = new EncryptionParameters();
	parms->set_poly_modulus("1x^4096 + 1");
  parms->set_coeff_modulus(coeff_modulus_128(4096));

   
    //parms->set_plain_modulus(4000000000);
  parms->set_plain_modulus(1<<20);


  context = new SEALContext(*parms);


  PublicKey public_key;
  public_key.load(pubfile);
  SecretKey secret_key;
  secret_key.load(secfile);

  keygen=new KeyGenerator(*context,secret_key,public_key);

  encryptor = new Encryptor(*context, public_key);
  decryptor = new Decryptor(*context,secret_key);
  evaluator = new Evaluator(*context);
  intencoder = new IntegerEncoder(context->plain_modulus(),3);
  fraencoder =  new FractionalEncoder(context->plain_modulus(), context->poly_modulus(), 64, 32, 3);


  ev_keys16 = new EvaluationKeys();
  ev_keys16->load(evalfile);

  pubfile.close();
  secfile.close();
  evalfile.close();
  
   
}


void delParameters(){
   delete ev_keys16;
   delete fraencoder;
   delete intencoder;
   delete evaluator;
   delete decryptor;
   delete encryptor;
   delete keygen;
   delete context;
   delete parms;
}
//Precondition: setParameters() or initFromKeys() must be called before
//Encrypt a normalized image(that is a vector of float) using Fractional encoder in a 3d ciphertext of dim zd,xd,yd
ciphertext3D encryptImage(vector<float> image, int zd, int xd, int yd){

  ciphertext3D encrypted_image(zd,ciphertext2D(xd,vector<Ciphertext>(yd)));
  
    for(int z=0;z<zd;z++)
      for(int i=0;i<xd;i++)
        for(int j=0;j<yd;j++){
            encryptor->encrypt(fraencoder->encode(image[i*xd+j]),encrypted_image[z][i][j]);          
            }

  cout << "Noise budget in fresh encrypted pixel: ";
  cout<< decryptor->invariant_noise_budget(encrypted_image[0][0][0]) << " bits"<<endl;

  return encrypted_image;

}
//Precondition: setParameters() or initFromKeys() must be called before
//Encrypt a normalized image(that is a vector of float) using Fractional encoder in a 3d ciphertext of dim zd,xd,yd and save it in file file_name
ciphertext3D encryptAndSaveImage(vector<float> image, int zd, int xd, int yd, string file_name){
  ofstream outfile(file_name, ofstream::binary);
  ciphertext3D encrypted_image(zd,ciphertext2D(xd,vector<Ciphertext>(yd)));
  
    for(int z=0;z<zd;z++)
      for(int i=0;i<xd;i++)
        for(int j=0;j<yd;j++){
            encryptor->encrypt(fraencoder->encode(image[i*xd+j]),encrypted_image[z][i][j]);
            encrypted_image[z][i][j].save(outfile);       
            }

  outfile.close();
  cout << "Noise budget in fresh encrypted pixel: ";
  cout<< decryptor->invariant_noise_budget(encrypted_image[0][0][0]) << " bits"<<endl;

  return encrypted_image;
}
//Precondition: initFromKeys() must be called before and encryptAndSaveImage() with that parameters must have been called before
//Load a normalized image saved it in file file_name in a 3d ciphertext of dim zd,xd,yd and 
ciphertext3D loadEncryptedImage(int zd, int xd, int yd, string file_name){
    ifstream imagefile(file_name, ifstream::binary);
    ciphertext3D encrypted_image(zd,ciphertext2D(xd,vector<Ciphertext>(yd)));

    for(int z=0;z<zd;z++)
      for(int i=0;i<xd;i++)
        for(int j=0;j<yd;j++){
            encrypted_image[z][i][j].load(imagefile);       
            }

    imagefile.close();
    return encrypted_image;
}
//Decrypt a fractional encrypted 3d image
floatCube decryptImage(ciphertext3D encrypted_image){
    int zd=encrypted_image.size();
    int xd=encrypted_image[0].size();
    int yd=encrypted_image[0][0].size();
    cout<<zd<<","<<xd<<","<<yd<<endl;

    Plaintext tmp;
    floatCube image(zd, vector<vector<float> > (xd,vector<float>(yd)));

    for(int z=0;z<zd;z++){
        cout<<z<<endl;
        for(int i=0;i<xd;i++){
            for(int j=0;j<yd;j++){
                    decryptor->decrypt(encrypted_image[z][i][j], tmp);
                    image[z][i][j]=fraencoder->decode(tmp);
                    cout<<image[z][i][j]<<" ";
        }
    cout<<endl;
    }
    cout<<endl;
    cout<<endl;
  }
  return image;
}


/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters()
{
    cout << "/ Encryption parameters:" << endl;
    cout << "| poly_modulus: " << context->poly_modulus().to_string() << endl;

    /*
    Print the size of the true (product) coefficient modulus
    */
    cout << "| coeff_modulus size: " 
        << context->total_coeff_modulus().significant_bit_count() << " bits" << endl;

    cout << "| plain_modulus: " << context->plain_modulus().value() << endl;
    cout << "\\ noise_standard_deviation: " << context->noise_standard_deviation() << endl;
    cout << endl;
}