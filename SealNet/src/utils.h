#ifndef UTILS
#define UTILS 

#include <vector>
using namespace std;
//normalizes a dataset (a vector of images) as img'=(img-mean)/std
vector<vector<float> > normalize(vector<vector<float> > dataset, float mean, float stdv);



#endif

