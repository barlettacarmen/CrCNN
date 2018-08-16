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
	setParameters();
	CnnBuilder build("PlainModelWoPad.h5");
	BatchNormLayer *bn=build.buildBatchNormLayer("pool1_features.norm1",20);
	cout<<bn->mean.size()<<endl;
	cout<<fraencoder->decode(bn->mean[0])<<endl;

	delParameters();

	return 0;
}
