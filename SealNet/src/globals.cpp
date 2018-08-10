#include "globals.h"
#include "seal/seal.h"
#include <fstream>
#include <string>

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
	parms = new EncryptionParameters();
	parms->set_poly_modulus("1x^4096 + 1");
  parms->set_coeff_modulus(coeff_modulus_128(4096));
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
    fraencoder =  new FractionalEncoder(context->plain_modulus(), context->poly_modulus(), 64, 32, 3);
   	ev_keys16 = new EvaluationKeys();
    keygen->generate_evaluation_keys(16, *ev_keys16);
   
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


/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters(const SEALContext &context)
{
    cout << "/ Encryption parameters:" << endl;
    cout << "| poly_modulus: " << context.poly_modulus().to_string() << endl;

    /*
    Print the size of the true (product) coefficient modulus
    */
    cout << "| coeff_modulus size: " 
        << context.total_coeff_modulus().significant_bit_count() << " bits" << endl;

    cout << "| plain_modulus: " << context.plain_modulus().value() << endl;
    cout << "\\ noise_standard_deviation: " << context.noise_standard_deviation() << endl;
    cout << endl;
}