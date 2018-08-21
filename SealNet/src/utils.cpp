#include "utils.h"
#include "mnist/mnist_reader.h"
#include <vector>
#include <string>


//The division by 255 is to have coerency with pythorch normalization of MNIST dataset
vector<vector<float> > normalize(vector<vector<float> > dataset, float mean, float stdv){

	for(int i=0;i<dataset.size();i++)
        	for(int j=0;j<dataset[0].size();j++)
        		dataset[i][j]=(dataset[i][j]/255 - mean)/stdv;

    return dataset;

}


vector<vector<float> > loadAndNormalizeMNISTestSet(string dataset_path){
	//Import MNIST dataset
    mnist::MNIST_dataset<vector, vector<float>, uint8_t> dataset =
    mnist::read_dataset<vector, vector, float, uint8_t>(dataset_path);
    //Normalize dataset (by data'=(data-mean)/std)
     

    dataset.test_images=normalize(dataset.test_images,0.1307, 0.3081);

    return dataset.test_images;
} 