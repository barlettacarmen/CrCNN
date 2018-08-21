#ifndef GLOBALS_H
#define GLOBALS_H

#include "seal/seal.h"
#include <vector>
#include <string>

using namespace seal;

typedef vector<vector<vector<Ciphertext> > > ciphertext3D;
typedef vector<vector<vector<Plaintext> > >  plaintext3D;
typedef vector<vector<Ciphertext> >	 ciphertext2D;
typedef vector<vector<Plaintext> >   plaintext2D;
typedef vector<vector<vector<vector<Plaintext> > > >  plaintext4D;
typedef vector<vector<vector<vector<float> > > >  floatHypercube;
typedef vector<vector<vector<float> > >   floatCube;

extern EncryptionParameters * parms;
extern KeyGenerator * keygen;
extern SEALContext * context;
extern Encryptor * encryptor;
extern Decryptor * decryptor;
extern Evaluator * evaluator;
extern IntegerEncoder *intencoder;
extern FractionalEncoder *fraencoder;
extern EvaluationKeys * ev_keys16;

void setParameters();
void setAndSaveParameters(string public_key_path,string secret_key_path,string evaluation_key_path);
void initFromKeys(string public_key_path,string secret_key_path,string evaluation_key_path);
void print_parameters();
void delParameters();
//Precondition: setParameters() or initFromKeys() must be called before
//Encrypt a normalized image(that is a vector of float) using Fractional encoder in a 3d ciphertext of dim zd,xd,yd
ciphertext3D encryptImage(vector<float> image, int zd, int xd, int yd);
//Precondition: setParameters() or initFromKeys() must be called before
//Encrypt a normalized image(that is a vector of float) using Fractional encoder in a 3d ciphertext of dim zd,xd,yd and save it in file file_name
ciphertext3D encryptAndSaveImage(vector<float> image, int zd, int xd, int yd, string file_name);
//Precondition: initFromKeys() must be called before and encryptAndSaveImage() with that parameters must have been called before
//Load a normalized image saved it in file file_name in a 3d ciphertext of dim zd,xd,yd and 
ciphertext3D loadEncryptedImage(int zd, int xd, int yd, string file_name);
//Decrypt a fractional encrypted 3d image
floatCube decryptImage(ciphertext3D encrypted_image);

#endif