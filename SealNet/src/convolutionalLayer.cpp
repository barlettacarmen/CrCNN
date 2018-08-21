#include "convolutionalLayer.h"
#include "layer.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>
#include <thread>
#include <ostream>
#include <fstream>

using namespace std;
using namespace seal;


ConvolutionalLayer::ConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf, int th_count,plaintext4D & filters, vector<Plaintext> & biases):
	Layer(name), xd(xd), yd(yd), zd(zd),
	xs(xs), ys(ys),
	xf(xf), yf(yf),
	nf(nf),
    /*Number of threads to use to split the computation*/
    th_count(th_count),
	/* Compute the output dimensions of the convolved ciphertext image.
	The function requires the dimensions of the image, the strides, (the padding) and
	the dimensions of the filters. The output is returned through xo,yo and zo. */
	xo( (xd-xf) / xs + 1 ), yo( (yd-yf) / ys + 1 ), zo(  nf),
	filters(filters),
    biases(biases){

}
//Initializes weighs and biases by loading them from file_name, where they are saved
ConvolutionalLayer::ConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf, int th_count,string file_name):
    Layer(name), xd(xd), yd(yd), zd(zd),
    xs(xs), ys(ys),
    xf(xf), yf(yf),
    nf(nf),
    th_count(th_count),
    /* Compute the output dimensions of the convolved ciphertext image.
    The function requires the dimensions of the image, the strides, (the padding) and
    the dimensions of the filters. The output is returned through xo,yo and zo. */
    xo( (xd-xf) / xs + 1 ), yo( (yd-yf) / ys + 1 ), zo(  nf)
    {
        loadPlaintextParameters(file_name);

}

    //da iterare sui 50 kernel
    ciphertext2D ConvolutionalLayer::convolution3d(ciphertext3D image, plaintext3D kernel, Plaintext bias){
        assert(kernel.size()==zd);
        assert(kernel[0].size()==xf);
        assert(kernel[0][0].size()==yf);
        
        int i,j,kx,ky,z,xlast,ylast,p;
        vector<Ciphertext> pixels(xf*yf*zd);
        ciphertext2D result(xo, vector<Ciphertext>(yo));

        Layer::computeBoundaries(xd,yd,xs,ys,xf,yf, &xlast, &ylast);

        for(i=0;i<xlast;i+=xs){
            assert(i<image[0].size());
            for(j=0;j<ylast;j+=ys){
                assert(j<image[0][0].size());
                p=0;
                for(z=0;z<zd;z++)
                    for(kx=0;kx<xf;kx++)
                        for(ky=0;ky<yf;ky++){
                                evaluator->multiply_plain(image[z][i+kx][j+ky],kernel[z][kx][ky],pixels[p],MemoryPoolHandle::Global());
                                p++;
            }
            //adding bias
            evaluator->add_plain(pixels[0],bias);
            evaluator->add_many(pixels,result[i/xs][j/ys]);
                        }
        }
        return result;

    }

//Forward with threads
ciphertext3D ConvolutionalLayer::forward (ciphertext3D input){
    int from=0,to=0,thread_kernels=0;
    ciphertext3D convolved(zo,ciphertext2D(xo,vector<Ciphertext>(yo)));
    vector<thread> th_vector;

    auto parallelForward=[&](ciphertext3D &input,ciphertext3D &convolved,int from, int to){
        for(int k=from;k<to;k++){
            convolved[k]=convolution3d(input,filters[k],biases[k]);
        }

    };

    if(th_count>nf)
        th_count=nf;

    thread_kernels=nf/th_count;
    
    
    for (int i = 0; i < th_count; i++){
        from=to;
        if(i<th_count-1)
            to+=thread_kernels;
        else
            to+=thread_kernels + (nf%th_count);

        th_vector.emplace_back(parallelForward, ref(input),ref(convolved),from,to);

    }
    for (size_t i = 0; i < th_vector.size(); i++)
    {
        th_vector[i].join();
    }

    return convolved;
}

//Forward without threads
/*
ciphertext3D ConvolutionalLayer::forward (ciphertext3D input){
    Plaintext bias;
	plaintext3D kernel(zd, plaintext2D(xf,vector<Plaintext>(yf)));
    ciphertext3D convolved(zo,ciphertext2D(xo,vector<Ciphertext>(yo)));
	for(int k=0;k<nf;k++){
		kernel=getKernel(k);
        bias=getBias(k);
        convolved[k]=convolution3d(input,kernel,bias);
	}
    return convolved;
}*/

void ConvolutionalLayer::savePlaintextParameters(string file_name){
    int i,j,z,n;
    ofstream outfile(file_name, ofstream::binary);
    for(n=0;n<nf;n++){
        for(z=0;z<zd;z++){
            for(i=0;i<xf;i++){
                for(j=0;j<yf;j++){
                    filters[n][z][i][j].save(outfile);
                }
            }
        }
        biases[n].save(outfile);
    }
    outfile.close();

}
void ConvolutionalLayer::loadPlaintextParameters(string file_name){
    ifstream infile(file_name, ifstream::binary);

    int n,z,i,j;
    plaintext4D encoded_weights(nf, plaintext3D(zd, plaintext2D (xf, vector<Plaintext> (yf ) )));
    vector<Plaintext> encoded_biases(nf);

    for(n=0;n<nf;n++){
        for(z=0;z<zd;z++){
            for(i=0;i<xf;i++){
                for(j=0;j<yf;j++){
                    encoded_weights[n][z][i][j].load(infile);
                }
            }
        }
        encoded_biases[n].load(infile);
    }
    infile.close();
    filters=encoded_weights;
    biases=encoded_biases;
}

plaintext3D ConvolutionalLayer::getKernel(int kernel_index){
    return filters[kernel_index];

}

Plaintext ConvolutionalLayer::getBias(int bias_index){
   return biases[bias_index]; 
 }

void ConvolutionalLayer::printLayerStructure(){
    cout<<"Convolutional "<<name<<" : input ("<<zd<<","<<xd<<","<<yd<<"); kernel("<<nf<<","<<xf<<","<<yf<<"); stride("<<xs<<","<<ys<<"); output("<<
    zo<<","<<xo<<","<<yo<<") "<<"run with "<<th_count<<" threads"<<endl;
}

ConvolutionalLayer::~ConvolutionalLayer(){}
