#include "convolutionalLayer.h"
#include "layer.h"
#include "ciphertextIO.h"
#include "seal/seal.h"
#include "globals.h"
#include <cassert>

using namespace std;
using namespace seal;


ConvolutionalLayer::ConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf, floatHypercube & filters, vector<float> & biases):
	Layer(name), xd(xd), yd(yd), zd(zd),
	xs(xs), ys(ys),
	xf(xf), yf(yf),
	nf(nf),
	/* Compute the output dimensions of the convolved ciphertext image.
	The function requires the dimensions of the image, the strides, the padding and
	the dimensions of the filters. The output is returned through xo,yo and zo. */
	xo( (xd-xf+2*xpad) / xs + 1 ), yo( (yd-yf+2*ypad) / ys + 1 ), zo(  nf),
	xpad(0),
	ypad(0),
	filters(filters),
    biases(biases){

}
//solo per prova
ConvolutionalLayer::ConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf):
    Layer(name), xd(xd), yd(yd), zd(zd),
    xs(xs), ys(ys),
    xf(xf), yf(yf),
    nf(nf),
    /* Compute the output dimensions of the convolved ciphertext image.
    The function requires the dimensions of the image, the strides, the padding and
    the dimensions of the filters. The output is returned through xo,yo and zo. */
    xo( (xd-xf+2*xpad) / xs + 1 ), yo( (yd-yf+2*ypad) / ys + 1 ), zo(  nf),
    xpad(0),
    ypad(0){

}

    //da iterare sui 50 kernel
    ciphertext2D ConvolutionalLayer::convolution3d(ciphertext3D image, plaintext3D kernel){
        assert(kernel.size()==zd);
        assert(kernel[0].size()==xf);
        assert(kernel[0][0].size()==yf);
        //bisogna aggiungere il bias
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
                                evaluator->multiply_plain(image[z][i+kx][j+ky],kernel[z][kx][ky],pixels[p]);
                                p++;
            }
            evaluator->add_many(pixels,result[i/xs][j/ys]);
                        }
        }
        return result;

    }

//da controllare assegnamento finale
ciphertext3D ConvolutionalLayer::forward (ciphertext3D input){
	plaintext3D kernel(zd, plaintext2D(xf,vector<Plaintext>(yf)));
    ciphertext3D convolved(zo,ciphertext2D(xo,vector<Ciphertext>(yo)));
	for(int k=0;k<nf;k++){
		kernel=getKernel(k);
        convolved[k]=convolution3d(input,kernel);
	}
    return convolved;
}

plaintext3D ConvolutionalLayer::getKernel(int kernel_index){return plaintext3D();}

ConvolutionalLayer::~ConvolutionalLayer(){}
