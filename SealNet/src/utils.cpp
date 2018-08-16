#include "utils.h"


//The division by 255 is to have coerency with pythorch normalization of MNIST dataset
vector<vector<float> > normalize(vector<vector<float> > dataset, float mean, float stdv){

	for(int i=0;i<dataset.size();i++)
        	for(int j=0;j<dataset[0].size();j++)
        		dataset[i][j]=(dataset[i][j]/255 - mean)/stdv;

    return dataset;

}