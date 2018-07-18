#include "globals.h"
#include "seal/seal.h"

using namespace seal;

EncryptionParameters *parms;
SEALContext * context;
KeyGenerator * keygen;
Encryptor * encryptor;
Decryptor * decryptor;
Evaluator * evaluator;
//PolyCRTBuilder * crtbuilder;

void setParameters(){
	parms = new EncryptionParameters();
	parms->set_poly_modulus("1x^8192 + 1");
    parms->set_coeff_modulus(coeff_modulus_128(8192));
    //parms->set_poly_modulus("1x^32768 + 1");
    //parms->set_coeff_modulus(coeff_modulus_128(32768));

    parms->set_plain_modulus(1099511922689);
    //parms->set_plain_modulus(1<<20);


    context = new SEALContext(*parms);
    KeyGenerator keygen(*context);
    auto public_key = keygen.public_key();
    auto secret_key= keygen.secret_key();
	encryptor = new Encryptor(*context, public_key);
	decryptor = new Decryptor(*context,secret_key);
  	evaluator = new Evaluator(*context);
   	//crtbuilder = new PolyCRTBuilder(*context);
   	cout<<"done"<<endl;
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