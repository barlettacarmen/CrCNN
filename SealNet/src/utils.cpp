#include "utils.h"



vector<vector<float> > normalize(vector<vector<float> > dataset, float mean, float stdv){

	for(int i=0;i<dataset.size();i++)
        	for(int j=0;j<dataset[0].size();j++)
        		dataset[i][j]=(dataset[i][j]-mean)/stdv;

    return dataset;

}