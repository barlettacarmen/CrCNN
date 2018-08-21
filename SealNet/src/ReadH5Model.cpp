#include <iostream>
#include <vector>
#include <string>
#include "cnnBuilder.h"
#include "H5Easy.h"

using namespace std;

int main()
{ //le norm non dovrebbero servire

   LoadH5 ldata;
   ldata.setFileName("PlainModelWoPad.h5");
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

   cout<<"var"<<endl;
   for ( vector<float>::iterator it = n1var.begin(); it != n1var.end(); ++it )
     cout << *it << endl;
   cout<<"mean"<<endl;
   for(int i=0;i<n1mean.size();i++)
  	cout<<n1mean[i] <<endl;
/*
   CnnBuilder b("PlainModel.h5");
   b.buildConvolutionalLayer("pool1_features.conv1",28,28,1,2,2,5,5,20);

*/
  
   return 0;
}
