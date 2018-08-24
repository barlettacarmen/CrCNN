#ifndef CNN_BUILDER
#define CNN_BUILDER
#include "H5Easy.h"
#include "convolutionalLayer.h"
#include "fullyConnectedLayer.h"
#include "poolingLayer.h"
#include "squareLayer.h"
#include "batchNormLayer.h"
#include "network.h"
#include <string>
#include <vector>
#include <ostream>
#include <fstream>

class CnnBuilder{

public:
	string plain_model_path;
	LoadH5 ldata;
	CnnBuilder(string plain_model_path);
	~CnnBuilder();
	vector<float> getPretrained(string var_name);
	ConvolutionalLayer * buildConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf, int th_count,istream * infile);
	FullyConnectedLayer * buildFullyConnectedLayer(string name, int in_dim, int out_dim, int th_count,istream * infile);
	PoolingLayer * buildPoolingLayer(string name,int xd,int yd, int zd, int xs, int ys,int xf, int yf);
	SquareLayer *  buildSquareLayer(string name, int th_count);
	BatchNormLayer * buildBatchNormLayer(string name, int num_channels,istream * infile);
	/* Define all necessary layers with their parameters in this function
	If the encoded Network has been already saved in file_name, then it should be the right path of encoded net*/
	Network buildNetwork(string file_name="");
	/*---------------------*/
	Network buildAndSaveNetwork(string file_name);
	//Loads the encoded network by calling the approptiate constructors
	

};








#endif
