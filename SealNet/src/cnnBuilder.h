#ifndef CNN_BUILDER
#define CNN_BUILDER
#include "H5Easy.h"
#include "convolutionalLayer.h"
#include "fullyConnectedLayer.h"
#include "poolingLayer.h"
#include "squareLayer.h"
#include "network.h"
#include <string>
#include <vector>

class CnnBuilder{

public:
	string plain_model_path;
	LoadH5 ldata;
	CnnBuilder(string plain_model_path);
	~CnnBuilder();
	vector<float> getPretrained(string var_name);
	ConvolutionalLayer * buildConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf);
	FullyConnectedLayer * buildFullyConnectedLayer(string name, int in_dim, int out_dim);
	PoolingLayer * buildPoolingLayer(string name,int xd,int yd, int zd, int xs, int ys,int xf, int yf);
	SquareLayer *  buildSquareLayer(string name);
	/* Define all necessary layers with their parameters in this function*/
	Network buildNetwork();

};








#endif
