#ifndef GLOBALS_H
#define GLOBALS_H

#include "seal/seal.h"
#include <vector>
using namespace seal;

typedef vector<vector<vector<Ciphertext> > > ciphertext3D;
typedef vector<vector<vector<Plaintext> > >  plaintext3D;
typedef vector<vector<Ciphertext> >	 ciphertext2D;
typedef vector<vector<Plaintext> >   plaintext2D;

extern EncryptionParameters * parms;
extern SEALContext * context;
extern Encryptor * encryptor;
extern Decryptor * decryptor;
extern Evaluator * evaluator;
//extern PolyCRTBuilder * crtbuilder;

void setParameters();

void delParameters();

#endif