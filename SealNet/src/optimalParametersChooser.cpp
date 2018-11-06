#include "optimalParametersChooser.h"
#include "globals.h"
#include "seal/seal.h"
#include "cnnBuilder.h"
#include "network.h"
#include "utils.h"
#include <vector>
#include <algorithm>
#include <math.h>
#include <cassert> 
#include <random>
#include <chrono>
using namespace std;
using namespace seal;

vector<vector<float> > test_set;
vector<unsigned char> predicted_labels;
unsigned seed=0;
int calls=0;

enum exit_status_forward{SUCCESS, OUT_OF_BUDGET,MISPREDICTED};

uint64_t plainModulusBinarySearchInternal(CnnBuilder build,uint64_t min_plain_modulus, uint64_t max_plain_modulus, int max_poly_modulus, bool pow, int num_images_to_test);
exit_status_forward testPlainModulus(CnnBuilder build, uint64_t plain_modulus,int max_poly_modulus, int num_images_to_test);
uint64_t minSmallModulusinCoeffModulus(int max_poly_modulus);


//path_to_model="PlainModelWoPad.h5"
uint64_t plainModulusBinarySearch(int num_images_to_test,uint64_t min_plain_modulus, uint64_t max_plain_modulus, string path_to_model){

	int max_poly_modulus=4096;
	bool pow=true;
	uint64_t plain_modulus;


	/*Load and normalize your dataset*/
	test_set=loadAndNormalizeMNISTestSet("../PlainModel/MNISTdata/raw");
	/*Load predictions of plain model*/
	predicted_labels=loadMNISTPlainModelPredictions("../PlainModel/predictionsApproxPlainModel.csv");
	
	CnnBuilder build(path_to_model);
	/*Search between powers of two*/
	plain_modulus=plainModulusBinarySearchInternal(build,min_plain_modulus,max_plain_modulus,max_poly_modulus,pow,num_images_to_test);
	/*Take the smallest SmallModulus betwen the prime numbers (qi) associated to the default 128 coeff_modulus parameter*/
	uint64_t min_prime_coeff_mod=minSmallModulusinCoeffModulus(max_poly_modulus);
	/*If a plain_modulus is found, but it is bigger than at least one of the qi, fast_plain_lift won't be enabled, and multiply_plain will be slower.
	To enable it all the small moduli {q1,q2,...,qk} which construct the coefficient modulus must be smaller than plaintext modulus t.
	If this is true, then Evaluator::multiply_plain becomes significantly faster.*/
	if(plain_modulus>0 && plain_modulus>=min_prime_coeff_mod){
		cout<<plain_modulus<<" found, but plain_lift is not enabled"<<endl;
		/*We want to find the min plain_mod s.t. fast_plain_lift is enabled, otherwise return the one found before.
		Search between all integers x s.t 2^(floor(log2(min_prime_coeff_mod))<= x <= min_prime_coeff_mod-1, 
		because in this way we have less possibility to decrease the plain_modulus, we have found so far, while the noise budget can only increase. */
		max_plain_modulus=min_prime_coeff_mod-1;
		min_plain_modulus= 1UL<<int(floor(log2(min_prime_coeff_mod)));
		cout<<"continue searching between "<<min_plain_modulus<<" and "<<max_plain_modulus<<endl;
		pow=false;
		uint64_t plain_modulus_fast=plainModulusBinarySearchInternal(build,min_plain_modulus,max_plain_modulus,max_poly_modulus,pow,num_images_to_test);
		if(plain_modulus_fast>0)
			return plain_modulus_fast;
	}
	return plain_modulus;

}

uint64_t minSmallModulusinCoeffModulus(int max_poly_modulus){
	vector<SmallModulus> q_primes=coeff_modulus_128(max_poly_modulus);
	uint64_t min=q_primes[0].value();
	for(int i=0; i<q_primes.size();i++){
		if(q_primes[i].value()<min)
			min=q_primes[i].value();
	}
	return min;
}

//Bool pow, indicates if we search between powers of two(pow=true)or not.
uint64_t plainModulusBinarySearchInternal(CnnBuilder build,uint64_t min_plain_modulus, uint64_t max_plain_modulus, int max_poly_modulus, bool pow, int num_images_to_test){
	uint64_t plain_modulus;
    assert(min_plain_modulus<=max_plain_modulus);
	cout<<"level "<<calls++<<"  ";
	assert(plain_modulus>=min_plain_modulus);
	exit_status_forward test_plain;
	/*If we are at the base of recursion i.e. we have scanned all intermediate integers numbers, we can try with the two extremes values and return
	the smaller of the two. If the min value returns OUT_OF_BUDGET we return 0, to signal that we haven't found any good plain_mod, and we don't test the 
	max value, since it will give us the same result, given that the noise budget will even be smaller. */
	if(pow){
		min_plain_modulus=log2(min_plain_modulus);
		max_plain_modulus=log2(max_plain_modulus);
	}
	if(max_plain_modulus-min_plain_modulus<=1){
		if(pow){
				min_plain_modulus=1UL<<min_plain_modulus;
				max_plain_modulus=1UL<<max_plain_modulus;
		}
		cout<<"!!base!!  ";
		cout<<"min_plain_modulus: "<<min_plain_modulus;
		cout<<"  max_plain_modulus: "<<max_plain_modulus<<endl;
		test_plain=testPlainModulus(build,min_plain_modulus,max_poly_modulus, num_images_to_test);
		cout<<"level "<<calls<<" returning ";
		if(test_plain==SUCCESS){
			calls--;
			cout<<min_plain_modulus<<" due to Success"<<endl;
			return min_plain_modulus;
		}
		if(test_plain==OUT_OF_BUDGET){
			calls--;
			cout<<"0 due to Out of Budget"<<endl;
			return 0;
		}
		if(pow){
			min_plain_modulus=log2(min_plain_modulus);
			max_plain_modulus=log2(max_plain_modulus);
		}
		if(max_plain_modulus-min_plain_modulus==1){
			if(pow){
				max_plain_modulus=1UL<<max_plain_modulus;
			}
			test_plain=testPlainModulus(build,max_plain_modulus,max_poly_modulus,num_images_to_test);
			if(test_plain==SUCCESS){
				calls--;
			    cout<<max_plain_modulus<< " due to Success"<<endl;
				return max_plain_modulus;
			}
			cout<<"0 due to Mispredict"<<endl;
			calls--;
			return 0;
		}
		else return 0;		
	}

	plain_modulus=min_plain_modulus+(max_plain_modulus-min_plain_modulus)/2;
	if(pow){
		plain_modulus=1UL<<plain_modulus;
		min_plain_modulus=1UL<<min_plain_modulus;
		max_plain_modulus=1UL<<max_plain_modulus;
	}

	cout<<"plain_modulus: "<<plain_modulus;
	cout<<"  min_plain_modulus: "<<min_plain_modulus;
	cout<<"  max_plain_modulus: "<<max_plain_modulus<<endl;

	test_plain=testPlainModulus(build,plain_modulus,max_poly_modulus,num_images_to_test);

	/*If SUCCESS we go left to try to find a smaller plain_mod, so if we find it we return it, otherwise we return the aldeady found plain mod*/
	/*If OUT_OF_BUDGET we go left to try with a smaller plain_mod, so to have a bigger noise budget. If we find a good enough plain_mod we return it,
	otherwise we return 0 to signal that we haven't found any good plain_mod */


	if(test_plain==SUCCESS||test_plain==OUT_OF_BUDGET){
		uint64_t recursion_result= plainModulusBinarySearchInternal(build,min_plain_modulus,plain_modulus-1,max_poly_modulus,pow,num_images_to_test);
		cout<<"level "<<calls<<" returning ";
		if(recursion_result>0){
			calls--;
			cout<<recursion_result<<" from lower level left"<<endl;
			return recursion_result;
		}
		if(test_plain==SUCCESS){
			calls--;
			cout<<plain_modulus<<" from himself"<<endl;
			return plain_modulus;
		}
		calls--;
		cout<<"0 since it failed, but also the lower level failed"<<endl;
		return 0;
	}

	/*If MISPREDICTED go right to find a bigger plain_mod */
	if(plain_modulus>=max_plain_modulus){
		cout<<"level "<<calls<<" returning ";
    	calls--;
	    cout<<"0 from lower base mispredicted"<<endl;
	    return 0;
	}
    auto ret_value = plainModulusBinarySearchInternal(build,plain_modulus+1,max_plain_modulus,max_poly_modulus,pow,num_images_to_test);
	cout<<"level "<<calls<<" returning ";
	calls--;
	cout<<ret_value<<" from lower level right"<<endl;
	return ret_value;

}



exit_status_forward testPlainModulus(CnnBuilder build, uint64_t plain_modulus,int max_poly_modulus, int num_images_to_test){

	default_random_engine generator(seed);
	uniform_int_distribution<int> distribution(0,predicted_labels.size());
	exit_status_forward ret_value=SUCCESS;

	cout<<"Testing "<<plain_modulus<<endl;
	
	setParameters(max_poly_modulus,plain_modulus);
	cout<<"keys done"<<endl<<flush;
	/*Encode the network  saved in path_to model*/
	Network net=build.buildNetwork();
	cout<<"Built network"<<endl;
	/*Encrypt and test "num_images_to_test"*/
	for(int i=0;i<num_images_to_test;i++){
		int img_test= distribution(generator);
		cout<<"Encrypting image "<<img_test<<" "<<flush;
		ciphertext3D encrypted_image=encryptImage(test_set[img_test], 1, 28, 28);
		cout<<"Testing image "<<img_test<<endl;
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
		cout<<"Comparing: predicted= "<<(uint32_t)predicted<<" plain res= "<<(uint32_t)predicted_labels[img_test]<<endl;
		if(predicted!=predicted_labels[img_test]){
			ret_value=MISPREDICTED;
			break;
			}
	}
	cout<< (ret_value == SUCCESS ? "Success" : (ret_value == OUT_OF_BUDGET ? "Out of Budget" : "Mispredicted") ) << endl;
	delParameters();
	return ret_value;

}


int main(){

	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_binary_search(0);
	uint64_t found_plain_mod;
	cout<<"Testing from 1UL<<24 to 1UL<<34 with POW"<<endl;
	for(int i=0;i<50;i++){
		cout<<"RUN "<<i<<endl;
		int rand=(i+1)*123476587+((i+2)*(i+3));
		cout<<"Random seed= num_images_to_test*"<<rand<<endl;
		for(int num_images_to_test=2;num_images_to_test<33;num_images_to_test=num_images_to_test<<1){
			seed=num_images_to_test*rand;
			cout<<"SEED = "<<seed<<endl;
			time_start = chrono::high_resolution_clock::now();
			found_plain_mod= plainModulusBinarySearch(num_images_to_test,1UL<<24,1UL<<34, "ApproxPlainModel.h5");
			time_end = chrono::high_resolution_clock::now();
			time_binary_search = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
			cout<<"OUTPUT:"<<num_images_to_test<<","<<found_plain_mod<<","<<time_binary_search.count()<<endl;
		}
	}
	
	return 0;
}
