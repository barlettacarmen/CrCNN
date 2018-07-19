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
//PolyCRTBuilder * crtbuilder;

void setParameters(){
	parms = new EncryptionParameters();
	parms->set_poly_modulus("1x^4096 + 1");
    parms->set_coeff_modulus(coeff_modulus_128(4096));
    //parms->set_poly_modulus("1x^32768 + 1");
    //parms->set_coeff_modulus(coeff_modulus_128(32768));

    //parms->set_plain_modulus(1099511922689);
    parms->set_plain_modulus(4000000000);
    //parms->set_plain_modulus(1<<20);


    context = new SEALContext(*parms);
    keygen= new KeyGenerator(*context);
    auto public_key = keygen->public_key();
    auto secret_key= keygen->secret_key();
	encryptor = new Encryptor(*context, public_key);
	decryptor = new Decryptor(*context,secret_key);
  	evaluator = new Evaluator(*context);
   	//crtbuilder = new PolyCRTBuilder(*context);
   
}


void initFromKeys(string key_pair_path){
	ifstream infile(key_pair_path, ifstream::binary);

	parms = new EncryptionParameters();
	parms->set_poly_modulus("1x^4096 + 1");
    parms->set_coeff_modulus(coeff_modulus_128(4096));

   
    parms->set_plain_modulus(4000000000);


    context = new SEALContext(*parms);
    

    PublicKey public_key;
    public_key.load(infile);
   	SecretKey secret_key;
   	secret_key.load(infile);

   	keygen=new KeyGenerator(*context,secret_key,public_key);

	encryptor = new Encryptor(*context, public_key);
	decryptor = new Decryptor(*context,secret_key);
  	evaluator = new Evaluator(*context);
   	//crtbuilder = new PolyCRTBuilder(*context);
   
}

void delParameters(){
  // delete crtbuilder;
   delete evaluator;
   delete decryptor;
   delete encryptor;
   delete keygen;
   delete context;
   delete parms;
}