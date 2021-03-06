#include "cnnBuilder.h"
#include "globals.h"
#include "convolutionalLayer.h"
#include "fullyConnectedLayer.h"
#include "network.h"
#include <string>
#include <vector>
#include <cmath>
#include <ostream>
#include <fstream>

using namespace std;

	CnnBuilder::CnnBuilder(string plain_model_path){
		
		ldata.setFileName(plain_model_path);

	}

	vector<float> CnnBuilder::getPretrained(string var_name){
		ldata.setVarName(var_name);
   		return ldata.getData();
	}

	ConvolutionalLayer * CnnBuilder::buildConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf, int th_count,istream * infile){
		if(infile!=NULL)
			return new ConvolutionalLayer(name,xd,yd,zd,xs,ys,xf,yf,nf,th_count,infile);

		int n,z,i,j,w;
		vector<float> weights,biases;
		plaintext4D encoded_weights(nf, plaintext3D(zd, plaintext2D (xf, vector<Plaintext> (yf ) )));
		vector<Plaintext> encoded_biases(nf);

		weights=getPretrained(name+".weight");
		biases=getPretrained(name+".bias");
		w=0;
		for(n=0;n<nf;n++){
			for(z=0;z<zd;z++)
				for(i=0;i<xf;i++)
					for(j=0;j<yf;j++){
						encoded_weights[n][z][i][j]=fraencoder->encode( weights[w]);
						//cout<<n<<" "<<z<<" "<<i<<" "<<j<<" -> "<<weights[w]<<endl;
						w++;
					}
			encoded_biases[n]=fraencoder->encode(biases[n]);
		}

		return new ConvolutionalLayer(name,xd,yd,zd,xs,ys,xf,yf,nf,th_count,encoded_weights,encoded_biases);

	}


	FullyConnectedLayer * CnnBuilder::buildFullyConnectedLayer(string name, int in_dim, int out_dim, int th_count,istream * infile){
		if(infile!=NULL)
			return new FullyConnectedLayer(name,in_dim,out_dim,th_count,infile);

		int i,j,w;
		vector<float> weights,biases;
		vector<Plaintext> encoded_biases(out_dim);
		plaintext2D encoded_weights(out_dim,vector<Plaintext> (in_dim));

		weights=getPretrained(name+".weight");
		biases=getPretrained(name+".bias");
		w=0;
		for(i=0;i<out_dim;i++){
			for(j=0;j<in_dim;j++){
				encoded_weights[i][j]=fraencoder->encode( weights[w] );
				//cout<<n<<" "<<z<<" "<<i<<" "<<j<<" -> "<<weights[w]<<endl;
				w++;
			}
			encoded_biases[i]=fraencoder->encode(biases[i]);
		}
	return new FullyConnectedLayer(name,in_dim,out_dim,th_count,encoded_weights,encoded_biases);

	}


	PoolingLayer * CnnBuilder::buildPoolingLayer(string name,int xd,int yd, int zd, int xs, int ys,int xf, int yf){
		return new PoolingLayer(name,xd,yd,zd,xs,ys,xf,yf);
	}
	AvgPoolingLayer * CnnBuilder::buildAvgPoolingLayer(string name,int xd,int yd, int zd, int xs, int ys,int xf, int yf){
		return new AvgPoolingLayer(name,xd,yd,zd,xs,ys,xf,yf);
	}
	SquareLayer *CnnBuilder::buildSquareLayer(string name, int th_count){
		return new SquareLayer(name,th_count);
	}
	
	//Encoding in plaintext mean and variance as 1/sqrt(var+1e-05)
	BatchNormLayer * CnnBuilder::buildBatchNormLayer(string name, int num_channels,istream * infile){
		if(infile!=NULL)
			return new BatchNormLayer(name,num_channels,infile);

		vector<float> mean,var;
		vector<Plaintext> encoded_mean(num_channels),encoded_var(num_channels);

		mean=getPretrained(name+".running_mean");
		var=getPretrained(name+".running_var");

		for(int i=0; i<num_channels;i++){
			encoded_mean[i]=fraencoder->encode(mean[i]);
			var[i]=1/sqrt(var[i] + 0.00001);
			encoded_var[i]=fraencoder->encode(var[i]);
		}
		return new BatchNormLayer(name,num_channels,encoded_mean,encoded_var);
	}


	Network CnnBuilder::buildNetwork(string file_name){
		int th_count=40,th_count2=50,th_tiny=32,th_tiny2=42;
		Network net;
		ifstream *infile=NULL;
		if(file_name!=""){
			infile = new ifstream(file_name, ifstream::binary);
		}
		//----ApproxPlainModel.h5-----
		/*ConvolutionalLayer *conv1 = buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20,th_count,infile);
		net.getLayers().push_back(shared_ptr<Layer> (conv1));
		AvgPoolingLayer *pool1 = buildAvgPoolingLayer("pool1",12,12,20,1,1,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool1));
		BatchNormLayer *bn1=buildBatchNormLayer("pool1_features.norm1",20,infile);
		net.getLayers().push_back(shared_ptr<Layer> (bn1));
		ConvolutionalLayer *conv2= buildConvolutionalLayer("pool2_features.conv2",11,11,20,2,2,3,3,50,th_count2,infile);
		net.getLayers().push_back(shared_ptr<Layer> (conv2));
		SquareLayer *act1= buildSquareLayer("act1",th_count2);
		net.getLayers().push_back(shared_ptr<Layer> (act1));
		AvgPoolingLayer *pool2= buildAvgPoolingLayer("pool2",5,5,50,1,1,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool2));
		BatchNormLayer *bn2=buildBatchNormLayer("pool2_features.norm2",50,infile);
		net.getLayers().push_back(shared_ptr<Layer> (bn2));
		FullyConnectedLayer *fc1= buildFullyConnectedLayer("classifier.fc3",4*4*50,500,th_count,infile);
		net.getLayers().push_back(shared_ptr<Layer> (fc1));
		FullyConnectedLayer *fc2= buildFullyConnectedLayer("classifier.fc4",500,10,th_count2,infile);
		net.getLayers().push_back(shared_ptr<Layer> (fc2));
		*/
		//----------------------------
		//----PlainNetWoPad.h5-------
		
		/*ConvolutionalLayer *conv1 = buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20,th_count,infile);
		net.getLayers().push_back(shared_ptr<Layer> (conv1));
		PoolingLayer *pool1 = buildPoolingLayer("pool1",12,12,20,1,1,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool1));
		BatchNormLayer *bn1=buildBatchNormLayer("pool1_features.norm1",20,infile);
		net.getLayers().push_back(shared_ptr<Layer> (bn1));
		ConvolutionalLayer *conv2= buildConvolutionalLayer("pool2_features.conv2",11,11,20,2,2,3,3,50,th_count,infile);
		net.getLayers().push_back(shared_ptr<Layer> (conv2));
		SquareLayer *act1= buildSquareLayer("act1",th_count);
		net.getLayers().push_back(shared_ptr<Layer> (act1));
		PoolingLayer *pool2= buildPoolingLayer("pool2",5,5,50,1,1,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool2));
		BatchNormLayer *bn2=buildBatchNormLayer("pool2_features.norm2",50,infile);
		net.getLayers().push_back(shared_ptr<Layer> (bn2));
		FullyConnectedLayer *fc1= buildFullyConnectedLayer("classifier.fc3",4*4*50,500,th_count,infile);
		net.getLayers().push_back(shared_ptr<Layer> (fc1));
		FullyConnectedLayer *fc2= buildFullyConnectedLayer("classifier.fc4",500,10,th_count,infile);
		net.getLayers().push_back(shared_ptr<Layer> (fc2));*/
		//------------------------------ 
		//----PlainNetTiny.h5---------
		ConvolutionalLayer *conv1 = buildConvolutionalLayer("pool1_features.conv1",28,28,1,1,1,5,5,32,th_tiny,infile);
		net.getLayers().push_back(shared_ptr<Layer> (conv1));
		AvgPoolingLayer *pool1 = buildAvgPoolingLayer("pool1",24,24,32,2,2,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool1));
		ConvolutionalLayer *conv2= buildConvolutionalLayer("pool2_features.conv2",12,12,32,1,1,5,5,64,th_tiny*2,infile);
		net.getLayers().push_back(shared_ptr<Layer> (conv2));
		AvgPoolingLayer *pool2= buildAvgPoolingLayer("pool2",8,8,64,2,2,2,2);
		net.getLayers().push_back(shared_ptr<Layer> (pool2));
		FullyConnectedLayer *fc1= buildFullyConnectedLayer("classifier.fc3",4*4*64,512,th_tiny2,infile);
		net.getLayers().push_back(shared_ptr<Layer> (fc1));
		FullyConnectedLayer *fc2= buildFullyConnectedLayer("classifier.fc4",512,10,th_tiny2,infile);
		net.getLayers().push_back(shared_ptr<Layer> (fc2));



		if(infile!=NULL){
			infile->close();
			delete infile;
		}

		return net;
	}
	//Precondition: setAndSaveParamters() or initFromKeys() must have been called before
	Network CnnBuilder::buildAndSaveNetwork(string file_name){
		ofstream * outfile = new ofstream(file_name, ofstream::binary);

		Network net=buildNetwork();
		for(int i=0; i<net.getNumLayers();i++){
			//cerr<<i<<endl<<flush;
			net.getLayer(i)->savePlaintextParameters(outfile);
		}

		outfile->close();

		delete outfile;

		return net;

	}

	/*---SIMULATOR-BUILDER---*/
	//Precondition=setChooserParameters() must have been called
	vector<ChooserPoly> CnnBuilder::buildSimulatedNetwork(int max_num_coefficients, int num_channels){

		vector<ChooserPoly> sim_input(num_channels);
		/*
	    First we create INPUT: a ChooserPoly representing the input data. You can think of 
	    this modeling a freshly encrypted ciphertext of a plaintext polynomial of
	    length at most max_num_coefficients, where the coefficients have absolute value 
	    at most 1 (as is the case when using IntegerEncoder with base 3).
	    */
		for(int i=0;i<num_channels;i++)
			sim_input[i]=ChooserPoly(max_num_coefficients,1);
		/*
		Then we simulate the forward by calling appropriate methods
		*/
		vector<float> w,b;
		sim_input=ConvolutionalLayer::convolutionalSimulator(sim_input,5,5, 32, w=getPretrained("pool1_features.conv1.weight"), b=getPretrained("pool1_features.conv1.bias"));
		sim_input=PoolingLayer::poolingSimulator(sim_input, 2, 2);
		//sim_input=BatchNormLayer::batchNormSimulator(sim_input,w=getPretrained("pool1_features.norm1.running_mean"),b=getPretrained("pool1_features.norm1.running_var"));
		sim_input=ConvolutionalLayer::convolutionalSimulator(sim_input,5,5, 64, w=getPretrained("pool2_features.conv2.weight"), b=getPretrained("pool2_features.conv2.bias"));
		//sim_input=SquareLayer::squareSimulator(sim_input);
		sim_input=PoolingLayer::poolingSimulator(sim_input, 2, 2);
		// sim_input=BatchNormLayer::batchNormSimulator(sim_input,w=getPretrained("pool2_features.norm2.running_mean"),b=getPretrained("pool2_features.norm2.running_var"));
		sim_input=FullyConnectedLayer::fullyConnectedSimulator(sim_input, w=getPretrained("classifier.fc3.weight"), b=getPretrained("classifier.fc3.bias"));
		sim_input=FullyConnectedLayer::fullyConnectedSimulator(sim_input, w=getPretrained("classifier.fc4.weight"), b=getPretrained("classifier.fc4.bias"));

		// ChooserPoly sim_input(10,1);
		// sim_input=ConvolutionalLayer::convolutionalSimulator(sim_input,5,5, 1);
		// sim_input=PoolingLayer::poolingSimulator(sim_input, 2, 2);
		// sim_input=BatchNormLayer::batchNormSimulator(sim_input);
		// sim_input=ConvolutionalLayer::convolutionalSimulator(sim_input,3,3, 20);
		// sim_input=SquareLayer::squareSimulator(sim_input);
		// sim_input=PoolingLayer::poolingSimulator(sim_input, 2, 2);
		// sim_input=BatchNormLayer::batchNormSimulator(sim_input);
		// sim_input=FullyConnectedLayer::fullyConnectedSimulator(sim_input,800);
		// sim_input=FullyConnectedLayer::fullyConnectedSimulator(sim_input, 500);


		return sim_input;


	}

	CnnBuilder::~CnnBuilder(){};