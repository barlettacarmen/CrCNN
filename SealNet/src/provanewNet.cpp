#include <vector>
#include <string>

#include "seal/seal.h"
#include "globals.h"
#include "network.h"
#include "cnnBuilder.h"
#include "utils.h"


using namespace std;
using namespace seal;

enum exit_status_forward{SUCCESS, OUT_OF_BUDGET,MISPREDICTED};
//Testing the Tiny network
int main(){
	
    /*Load and normalize your dataset*/
    vector<vector<float> > test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
    /*Load labels*/
    vector<unsigned char> labels=loadMNISTestLabels("../PlainModel/MNISTdata/raw");

    int num_images_to_test=1;
    exit_status_forward ret_value=SUCCESS;
    cout<<"Testing "<<endl;
    
    setParameters(2048,1UL<<19);
    cout<<"keys done"<<endl<<flush;
    /*Encode the network  saved in path_to model*/
    CnnBuilder build("PlainModelTiny.h5");
    Network net=build.buildNetwork();
    cout<<"Built network"<<endl;
    net.printNetworkStructure();
    /*Encrypt and test "num_images_to_test"*/
    for(int i=0;i<num_images_to_test;i++){
        cout<<"Encrypting image "<<i<<" ";
        ciphertext3D encrypted_image=encryptImage(test_set[i], 1, 28, 28);
        cout<<"Testing image "<<i<<endl;
        try{
            encrypted_image = net.forward(encrypted_image);
            }
        catch(OutOfBudgetException& e){
            /* If a reencryption has already been performed*/
            cout<<"Maximum layer computed is "<<e.last_layer_computed<<" exit due to OUT_OF_BUDGET"<<endl;
            ret_value=OUT_OF_BUDGET;
            break;
        }
        floatCube image=decryptImage(encrypted_image);

        auto it = max_element(begin(image[0]), end(image[0]));
        auto predicted = it - image[0].begin();
        /*If one of the tested images returns a wrong prediction, we need to find new plain_modulus*/
        if(predicted!=labels[i]){
            ret_value=MISPREDICTED;
            break;
            }
    }
    cout<< (ret_value == SUCCESS ? "Success" : (ret_value == OUT_OF_BUDGET ? "Out of Budget" : "Mispredicted") ) << endl;
    delParameters();
    //return ret_value;

}