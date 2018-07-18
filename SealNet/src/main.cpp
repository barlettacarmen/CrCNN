#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <random>
#include <limits>

#include "seal/seal.h"
#include "mnist/mnist_reader.h"

using namespace std;
using namespace seal;


/*
Helper function: Prints the parameters in a SEALContext.
*/
void print_parameters(const SEALContext &context)
{
    cout << "/ Encryption parameters:" << endl;
    cout << "| poly_modulus: " << context.poly_modulus().to_string() << endl;

    /*
    Print the size of the true (product) coefficient modulus
    */
    cout << "| coeff_modulus size: " 
        << context.total_coeff_modulus().significant_bit_count() << " bits" << endl;

    cout << "| plain_modulus: " << context.plain_modulus().value() << endl;
    cout << "\\ noise_standard_deviation: " << context.noise_standard_deviation() << endl;
    cout << endl;
}

int main()
{ //Import MNIST
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
    mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("/Users/carmen/Desktop/UNI/Tesi/Tools/Pytorch/MNISTdata/raw");

    cout << "Nbr of training images = " << dataset.training_images.size() << endl;
    cout << "Nbr of training labels = " << dataset.training_labels.size() << endl;
    cout << "Nbr of test images = " << dataset.test_images.size() << endl;
    cout << "Nbr of test labels = " << dataset.test_labels.size() << endl;
    cout<<"'"<<(short)dataset.test_images[0][1]<<""<< endl;





    /*
    In this fundamental example we discuss and demonstrate a powerful technique 
    called `batching'. If N denotes the degree of the polynomial modulus, and T
    the plaintext modulus, then batching is automatically enabled in SEAL if
    T is a prime and congruent to 1 modulo 2*N. In batching the plaintexts are
    viewed as matrices of size 2-by-(N/2) with each element an integer modulo T.
    */
    EncryptionParameters parms;

    parms.set_poly_modulus("1x^8192 + 1");
    parms.set_coeff_modulus(coeff_modulus_128(8192));

    /*
    Note that 40961 is a prime number and 2*1024 divides 40960.

    t| primo e P % 2*1024 = 1
    t = 12289
    */
    parms.set_plain_modulus(1099511922689);

    SEALContext context(parms);
    print_parameters(context);

    /*
    We can see that batching is indeed enabled by looking at the encryption
    parameter qualifiers created by SEALContext.
    */
    auto qualifiers = context.qualifiers();
    cout << "Batching enabled: " << boolalpha << qualifiers.enable_batching << endl;
    cout <<"NTT enabled: " <<boolalpha <<qualifiers.enable_ntt <<endl;
    cout <<"Fast pain lift enabled: " <<boolalpha <<qualifiers.enable_fast_plain_lift <<endl;

    KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();

    /*
    We need to create Galois keys for performing matrix row and column rotations.
    Like evaluation keys, the behavior of Galois keys depends on a decomposition
    bit count. The noise budget consumption behavior of matrix row and column 
    rotations is exactly like that of relinearization. Thus, we refer the reader
    to example_basics_ii() for more details.

    Here we use a moderate size decomposition bit count.
    */
    GaloisKeys gal_keys;
    keygen.generate_galois_keys(30, gal_keys);

    /*
    Since we are going to do some multiplications we will also relinearize.
    */
    EvaluationKeys ev_keys;
    keygen.generate_evaluation_keys(30, ev_keys);

    /*
    We also set up an Encryptor, Evaluator, and Decryptor here.
    */
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    /*
    Batching is done through an instance of the PolyCRTBuilder class so need
    to start by constructing one.
    */
    PolyCRTBuilder crtbuilder(context);

    /*
    The total number of batching `slots' is degree(poly_modulus). The matrices 
    we encrypt are of size 2-by-(slot_count / 2).
    */
    int slot_count = crtbuilder.slot_count();
    int row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    /*
    Printing the matrix is a bit of a pain.
    */
    auto print_matrix = [row_size](const vector<uint64_t> &matrix)
    {
        cout << endl;

        /*
        We're not going to print every column of the matrix (there are 2048). Instead
        print this many slots from beginning and end of the matrix.
        */
        int print_size = 5;

        cout << "    [";
        for (int i = 0; i < print_size; i++)
        {
            cout << setw(3) << matrix[i] << ",";
        }
        cout << setw(3) << " ...,";
        for (int i = row_size - print_size; i < row_size; i++)
        {
            cout << setw(3) << matrix[i] << ((i != row_size - 1) ? "," : " ]\n");
        }
        cout << "    [";
        for (int i = row_size; i < row_size + print_size; i++)
        {
            cout << setw(3) << matrix[i] << ",";
        }
        cout << setw(3) << " ...,";
        for (int i = 2 * row_size - print_size; i < 2 * row_size; i++)
        {
            cout << setw(3) << matrix[i] << ((i != 2 * row_size - 1) ? "," : " ]\n");
        }
        cout << endl;
    };

    /*
    The matrix plaintext is simply given to PolyCRTBuilder as a flattened vector
    of numbers of size slot_count. The first row_size numbers form the first row, 
    and the rest form the second row. Here we create the following matrix:

        [ 0,  1,  2,  3,  0,  0, ...,  0 ]
        [ 4,  5,  6,  7,  0,  0, ...,  0 ]
    */

    vector<uint64_t> pod_matrix(slot_count, 0);
    for(int i=0;i<10;i++){
        for(int j=0;j<28*28;j++){
            //pod_matrix[i*28*28+j]=dataset.test_images[i][j];
            pod_matrix[i*28*28+j]=i+j;

        }
    }

    cout << "Input plaintext matrix:" << endl;
    print_matrix(pod_matrix);

    /*
    First we use PolyCRTBuilder to compose the matrix into a plaintext.
    */
    Plaintext plain_matrix;
    crtbuilder.compose(pod_matrix, plain_matrix);

    /*
    Next we encrypt the plaintext as usual.
    */
    Ciphertext encrypted_matrix;
    cout << "Encrypting: ";
    encryptor.encrypt(plain_matrix, encrypted_matrix);
    cout << "Done" << endl;
    cout<<encrypted_matrix[0]<<endl;
    cout << "Noise budget in fresh encryption: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;




    /*
    Operating on the ciphertext results in homomorphic operations being performed
    simultaneously in all 4096 slots (matrix elements). To illustrate this, we 
    form another plaintext matrix

        [ 1,  2,  1,  2,  1,  2, ..., 2 ]
        [ 1,  2,  1,  2,  1,  2, ..., 2 ]

    and compose it into a plaintext.
    */
    FractionalEncoder encoder(context.plain_modulus(), context.poly_modulus(), 64, 32, 3);
    vector<double> pod_matrix2;
    for (int i = 0; i < slot_count; i++)
    {
        pod_matrix2.push_back((i % 2) + 100.1);
    }
    Plaintext plain_matrix2;
    vector<Plaintext> encoded_coefficients;
    cout << "Encoding plaintext coefficients: ";
    for (int i = 0; i < slot_count; i++)
    {
        encoded_coefficients.emplace_back(encoder.encode(pod_matrix2[i]));
    }

    //crtbuilder.compose(pod_matrix2, plain_matrix2);
    //cout << "Second input plaintext matrix:" << endl;
    //print_matrix(pod_matrix2);

    /*
    We now add the second (plaintext) matrix to the encrypted one using another 
    new operation -- plain addition -- and square the sum.
    */
    cout << "MUl and squaring: ";
    evaluator.multiply_plain(encrypted_matrix, plain_matrix2);
    evaluator.square(encrypted_matrix);
    evaluator.relinearize(encrypted_matrix, ev_keys);
    cout << "Done" << endl;

    /*
    How much noise budget do we have left?
    */
    cout << "Noise budget in result: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;
    
    /*
    We decrypt and decompose the plaintext to recover the result as a matrix.
    */
    Plaintext plain_result;
    cout << "Decrypting result: ";
    decryptor.decrypt(encrypted_matrix, plain_result);
    cout << "Done" << endl;

    vector<uint64_t> pod_result;
    crtbuilder.decompose(plain_result, pod_result);

    cout << "Result plaintext matrix:" << endl;
    print_matrix(pod_result);

    /*
    Note how the operation was performed in one go for each of the elements of the 
    matrix. It is possible to achieve incredible performance improvements by using 
    this method when the computation is easily vectorizable.

    All of our discussion so far could have applied just as well for a simple vector
    data type (not matrix). Now we show how the matrix view of the plaintext can be 
    used for more functionality. Namely, it is possible to rotate the matrix rows
    cyclically, and same for the columns (i.e. swap the two rows). For this we need
    the Galois keys that we generated earlier.

    We return to the original matrix that we started with.
    */
    encryptor.encrypt(plain_matrix, encrypted_matrix);
    cout << "Unrotated matrix: " << endl;
    print_matrix(pod_matrix);
    cout << "Noise budget in fresh encryption: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;

    /*
    Now rotate the rows to the left 3 steps, decrypt, decompose, and print.
    */
    evaluator.rotate_rows(encrypted_matrix, 3, gal_keys);
    cout << "Rotated rows 3 steps left: " << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    crtbuilder.decompose(plain_result, pod_result);
    print_matrix(pod_result);
    cout << "Noise budget after rotation: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;

    /*
    Rotate columns (swap rows), decrypt, decompose, and print.
    */
    evaluator.rotate_columns(encrypted_matrix, gal_keys);
    cout << "Rotated columns: " << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    crtbuilder.decompose(plain_result, pod_result);
    print_matrix(pod_result);
    cout << "Noise budget after rotation: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;

    /*
    Rotate rows to the right 4 steps, decrypt, decompose, and print.
    */
    evaluator.rotate_rows(encrypted_matrix, -4, gal_keys);
    cout << "Rotated rows 4 steps right: " << endl;
    decryptor.decrypt(encrypted_matrix, plain_result);
    crtbuilder.decompose(plain_result, pod_result);
    print_matrix(pod_result);
    cout << "Noise budget after rotation: "
        << decryptor.invariant_noise_budget(encrypted_matrix) << " bits" << endl;

    /*
    The output is as expected. Note how the noise budget gets a big hit in the
    first rotation, but remains almost unchanged in the next rotations. This is 
    again the same phenomenon that occurs with relinearization, where the noise 
    budget is consumed down to some bound determined by the decomposition bit count 
    and the encryption parameters. For example, after some multiplications have 
    been performed, rotations might practically for free (noise budget-wise), but
    might be relatively expensive when the noise budget is nearly full, unless
    a small decomposition bit count is used, which again is computationally costly.
    */
}