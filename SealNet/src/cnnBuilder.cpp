#include "cnnBuilder.h"
#include "globals.h"
#include "convolutionalLayer.h"
#include <string>
#include <vector>

using namespace std;

	CnnBuilder::CnnBuilder(string plain_model_path){
		
		ldata.setFileName(plain_model_path);

	}

	vector<float> CnnBuilder::getPretrained(string var_name){
		ldata.setVarName(var_name);
   		return ldata.getData();
	}

	ConvolutionalLayer CnnBuilder::buildConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf){
		int n,z,i,j,w;
		vector<float> weights,biases;
		floatHypercube reshaped_weights(nf, vector<vector<vector<float> > >(zd, vector<vector<float> > (xf, vector<float> (yf ) )));

		weights=getPretrained(name+".weight");
		biases=getPretrained(name+".bias");
		w=0;
		for(n=0;n<nf;n++)
			for(z=0;z<zd;z++)
				for(i=0;i<xf;i++)
					for(j=0;j<yf;j++){
						reshaped_weights[n][z][i][j]=weights[w];
						cout<<n<<" "<<z<<" "<<i<<" "<<j<<" -> "<<weights[w]<<endl;
						w++;
					}


		return ConvolutionalLayer(name,xd,yd,zd,xs,ys,xf,yf,nf,reshaped_weights,biases);



	}

	CnnBuilder::~CnnBuilder(){};