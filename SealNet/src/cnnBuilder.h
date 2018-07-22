#ifndef CNN_BUILDER
#define CNN_BUILDER
#include "H5Easy.h"
#include "convolutionalLayer.h"
#include <string>
#include <vector>

class CnnBuilder{

public:
	string plain_model_path;
	LoadH5 ldata;
	CnnBuilder(string plain_model_path);
	~CnnBuilder();
	vector<float> getPretrained(string var_name);
	ConvolutionalLayer buildConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf);

};








#endif
