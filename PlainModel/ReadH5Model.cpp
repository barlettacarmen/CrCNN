#include <iostream>
#include <vector>
#include <string>

#include "H5Easy.h"

using namespace std;

int main()
{ //le norm non dovrebbero servire

   LoadH5 ldata;
   ldata.setFileName("PlainModel.h5");
   ldata.setVarName("classifier.fc3.bias");
   vector<float>f3bias = ldata.getData();
   ldata.setVarName("classifier.fc3.weight");
   vector<float>f3weight = ldata.getData();
   ldata.setVarName("classifier.fc4.bias");
   vector<float>f4bias = ldata.getData();
   ldata.setVarName("classifier.fc4.weight");
   vector<float>f4weight = ldata.getData();
   ldata.setVarName("pool1_features.conv1.bias");
   vector<float>c1bias = ldata.getData();
   ldata.setVarName("pool1_features.conv1.weight");
   vector<float>c1weight = ldata.getData();
   ldata.setVarName("pool1_features.norm1.running_mean");
   vector<float>n1mean = ldata.getData();
   ldata.setVarName("pool1_features.norm1.running_var");
   vector<float>n1var = ldata.getData();
   ldata.setVarName("pool2_features.conv2.bias");
   vector<float>c2bias = ldata.getData();
   ldata.setVarName("pool2_features.conv2.weight");
   vector<float>c2weight = ldata.getData();
   ldata.setVarName("pool2_features.norm2.running_mean");
   vector<float>n2mean = ldata.getData();
   ldata.setVarName("pool2_features.norm2.running_var");
   vector<float>n2var = ldata.getData();

   //for ( vector<float>::iterator it = fdta.begin(); it != fdta.end(); ++it )
   //   cout << *it << endl;
   for(int i=0;i<c2bias.size();i++)
  	cout<< c2bias[i] <<endl;

  
   return 0;
}
