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
#include <algorithm>

#include "mnist/mnist_reader.h"
#include "seal/seal.h"
#include "globals.h"
#include "utils.h"
#include "cnnBuilder.h"
#include "batchNormLayer.h"
#include "H5Easy.h"


int main(int argc, char const *argv[])
{
	Plaintext before,after;
	initFromKeys("pub_key.txt","sec_key.txt","eval_key.txt");
	CnnBuilder build("PlainModelWoPad.h5");
	BatchNormLayer *bn1=build.buildBatchNormLayer("pool1_features.norm1",20);
	bn1->savePlaintextParameters("bn_params.txt");
	BatchNormLayer *bn2=build.buildBatchNormLayer("pool1_features.norm1",20,"bn_params.txt");

	for(int i=0;i<20;i++){
		before=bn1->getMean(i);
		after=bn2->getMean(i);
		cout<<"before mean"<<fraencoder->decode(before)<<endl<<flush;
		cout<<"after mean "<<fraencoder->decode(after)<<endl<<flush;
		before=bn1->getVar(i);
		after=bn2->getVar(i);
		cout<<"before var"<<fraencoder->decode(before)<<endl<<flush;
		cout<<"after var "<<fraencoder->decode(after)<<endl<<flush;

	}

	delParameters();

	return 0;
}
