#include "convolutionalLayer.h"
#include "layer.h"
#include "ciphertextIO.h"
#include "seal/seal.h"
#include "globals.h"

using namespace std;
using namespace seal;


ConvolutionalLayer::ConvolutionalLayer(string name,int xd,int yd,int zd,int xs,int ys,int xf,int yf,int nf,vector<float> filters):
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
	filters(filters){

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
        //bisogna aggiungere il bias
        int i,j,kx,ky,z,xlast,ylast,p;
        vector<Ciphertext> pixels(xf*yf*zd);
        ciphertext2D result(xo, vector<Ciphertext>(yo));

        Layer::computeBoundaries(xd,yd,xs,ys,xf,yf, &xlast, &ylast);

        for(i=0;i<xlast;i+=xs)
            for(j=0;j<ylast;j+=ys){
                p=0;
                for(kx=0;kx<xf;kx++)
                    for(ky=0;ky<yf;ky++)
                        for(z=0;z<zd;z++){
                            //evaluator->multiply_plain(image[i+kx][j+ky][z],kernel[kx][ky][z],pixels[(kx*xf*yf)+(ky*yf)+z]);
                            evaluator->multiply_plain(image[i+kx][j+ky][z],kernel[kx][ky][z],pixels[p]);
                            p++;
            }
            evaluator->add_many(pixels,result[i/xs][j/ys]);
                        }

        return result;

    }

//da controllae assegnamento finale
ciphertext3D ConvolutionalLayer::forward (ciphertext3D input){
	plaintext3D kernel(xf, plaintext2D(yf,vector<Plaintext>(zd)));
    ciphertext3D convolved(xo,ciphertext2D(yo,vector<Ciphertext>(zo)));
	for(int k=0;k<nf;k++){
		kernel=getKernel(k);
        convolved[k]=convolution3d(input,kernel);
	}
    return convolved;
}

plaintext3D ConvolutionalLayer::getKernel(int kernel_index){return plaintext3D();}

ConvolutionalLayer::~ConvolutionalLayer(){}
