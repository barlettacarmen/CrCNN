#include "cnnBuilder.h"
#include "globals.h"
#include "convolutionalLayer.h"
#include "fullyConnectedLayer.h"
#include "network.h"
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

	ConvolutionalLayer * CnnBuilder::buildConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf){
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
						//cout<<n<<" "<<z<<" "<<i<<" "<<j<<" -> "<<weights[w]<<endl;
						w++;
					}


		return new ConvolutionalLayer(name,xd,yd,zd,xs,ys,xf,yf,nf,reshaped_weights,biases);



	}

	FullyConnectedLayer * CnnBuilder::buildFullyConnectedLayer(string name, int in_dim, int out_dim){
		int i,j,w;
		vector<float> weights,biases;
		vector<vector<float> > reshaped_weights(out_dim, vector<float> (in_dim) );

		weights=getPretrained(name+".weight");
		biases=getPretrained(name+".bias");
		w=0;
		for(i=0;i<out_dim;i++)
			for(j=0;j<in_dim;j++){
				reshaped_weights[i][j]=weights[w];
				//cout<<n<<" "<<z<<" "<<i<<" "<<j<<" -> "<<weights[w]<<endl;
				w++;
			}
	return new FullyConnectedLayer(name,in_dim,out_dim,reshaped_weights,biases);

	}

	PoolingLayer * CnnBuilder::buildPoolingLayer(string name,int xd,int yd, int zd, int xs, int ys,int xf, int yf){
		return new PoolingLayer(name,xd,yd,zd,xs,ys,xf,yf);
	}
	SquareLayer *CnnBuilder::buildSquareLayer(string name){
		return new SquareLayer(name);
	}


	Network CnnBuilder::buildNetwork(){

		Network net;

		ConvolutionalLayer *conv1 = buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20);
		net.getLayers().push_back(shared_ptr<Layer> (conv1));
		PoolingLayer *pool1 = buildPoolingLayer("pool1",12,12,20,1,1,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool1));
		ConvolutionalLayer *conv2= buildConvolutionalLayer("pool2_features.conv2",11,11,20,2,2,3,3,50);
		net.getLayers().push_back(shared_ptr<Layer> (conv2));
		SquareLayer *act1= buildSquareLayer("act1");
		net.getLayers().push_back(shared_ptr<Layer> (act1));
		PoolingLayer *pool2= buildPoolingLayer("pool2",5,5,50,1,1,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool2));
		FullyConnectedLayer *fc1= buildFullyConnectedLayer("classifier.fc3",4*4*50,500);
		net.getLayers().push_back(shared_ptr<Layer> (fc1));
		FullyConnectedLayer *fc2= buildFullyConnectedLayer("classifier.fc4",500,10);
		net.getLayers().push_back(shared_ptr<Layer> (fc2));
		//net.printNetworkStructure();

		return net;
	}

	CnnBuilder::~CnnBuilder(){};