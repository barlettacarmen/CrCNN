#include "poolingLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"

using namespace std;
using namespace seal;


PoolingLayer:: PoolingLayer(string name,int xd,int yd, int zd, int xs, int ys,int xf, int yf):
	Layer(name),
	xd(xd), yd(yd),
	xs(xs), ys(ys),
	xf(xf), yf(yf),
	/* Compute the output dimensions of the pooled ciphertext image.
	The function requires the dimensions of the image, the strides and
	the dimensions of the filters. The output is returned through xo,yo. */
	xo((xd-xf)/xs + 1), yo((yd-yf)/ys + 1), zo(zd){
	}


ciphertext3D PoolingLayer::forward (ciphertext3D input){
	int i,j,kx,ky,z,xlast,ylast,p;
	vector<Ciphertext> pixels(xf*yf);
	ciphertext3D pooled_result(zo,ciphertext2D(xo,vector<Ciphertext>(yo)));

	Layer::computeBoundaries(xd,yd,xs,ys,xf,yf, &xlast, &ylast);

	for(z=0; z<zo; z++){
		for(i=0; i<xlast; i+=xs)
			for(j=0; j<ylast; j+=ys){
				p=0;
				for(kx=0; kx<xf; kx++)
					for(ky=0; ky<yf; ky++){
						pixels[p]=Ciphertext(input[z][i+kx][j+ky]);
						p++;
					}

				evaluator->add_many(pixels,pooled_result[z][i/xs][j/ys]);
			}
	}

	return pooled_result;
}

 void PoolingLayer::printLayerStructure(){
    cerr<<"Pooling "<<name<<" : input ("<<zo<<","<<xd<<","<<yd<<"); kernel("<<xf<<","<<yf<<"); stride("<<xs<<","<<ys<<"); output("<<
    zo<<","<<xo<<","<<yo<<")"<<endl;

}

PoolingLayer::~PoolingLayer(){}
