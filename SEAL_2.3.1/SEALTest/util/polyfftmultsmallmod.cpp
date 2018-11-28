#include "CppUnitTest.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/polyfftmultsmallmod.h"
#include "seal/util/smallntt.h"
#include "seal/util/mempool.h"
#include "seal/util/uintcore.h"
#include "seal/util/polycore.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/bigpoly.h"
#include "seal/bigpolyarray.h"
#include <random>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace seal;
using namespace seal::util;
using namespace std;


namespace SEALTest
{
    namespace util
    {
        TEST_CLASS(PolyFFTMultSmallMod)
        {
        public:
            TEST_METHOD(SmallNTTMultiplyPolyPoly)
            {
                MemoryPoolHandle pool = MemoryPoolHandle::Global();
                SmallModulus mod(4611686018427289601);
                int coeff_count_power = 2;
                SmallNTTTables tables;
                tables.generate(coeff_count_power, mod);

                Pointer poly1(allocate_poly(4, 1, pool));
                Pointer poly2(allocate_poly(4, 1, pool));
                Pointer result(allocate_poly(4, 1, pool));

                poly1[0] = 4611686018427289600;
                poly1[1] = 4611686018427289500;
                poly1[2] = 0;
                poly1[3] = 0;
                poly2[0] = 46116860;
                poly2[1] = 46116860;
                poly2[2] = 0;
                poly2[3] = 0;

                ntt_multiply_poly_poly(poly1.get(), poly2.get(), tables, result.get(), pool);
                Assert::AreEqual(4611686018381172741ULL, result.get()[0]);
                Assert::AreEqual(4611686013723369881ULL, result.get()[1]);
                Assert::AreEqual(4611686013769486741ULL, result.get()[2]);
                Assert::AreEqual(0ULL, result.get()[3]);

                coeff_count_power = 10;
                int coeff_count = (1 << coeff_count_power) + 1;
                poly1 = allocate_zero_poly(coeff_count, 1, pool);
                poly2 = allocate_zero_poly(coeff_count, 1, pool);
                result = allocate_zero_poly(coeff_count, 1, pool);
                Pointer correct(allocate_zero_poly(coeff_count, 1, pool));
                random_device rd;
                Pointer polymod(allocate_zero_poly(coeff_count, 1, pool));
                polymod[0] = 1;
                polymod[1024] = 1;
                PolyModulus polym(polymod.get(), coeff_count, 1);
                tables = SmallNTTTables(coeff_count_power, mod);

                for (int i = 0; i < coeff_count - 1; i++)
                {
                    poly1[i] = rd() % mod.value();
                    poly2[i] = rd() % mod.value();
                }

                nonfft_multiply_poly_poly_polymod_coeffmod(poly1.get(), poly2.get(), polym, mod, correct.get(), pool);
                ntt_multiply_poly_poly(poly1.get(), poly2.get(), tables, result.get(), pool);
                for (int i = 0; i < coeff_count; i++)
                {
                    Assert::AreEqual(correct[i], result[i]);
                }
            }

            TEST_METHOD(SmallNTTDotProductBigPolyArrayNTTBigPolyArray)
            {
                MemoryPoolHandle pool = MemoryPoolHandle::Global();
                SmallNTTTables tables;
                int coeff_uint64_count = divide_round_up(7, bits_per_uint64);

                BigPoly poly_modulus("1x^4 + 1");
                PolyModulus polymod(poly_modulus.data(), 5, coeff_uint64_count);
                SmallModulus mod(97);
                BigPoly result(5, 7);

                tables.generate(2, mod);

                // testing with general BigPolyArray and zero BigPolyArray
                BigPolyArray testzero_arr1(3, 5, 7);
                BigPolyArray testzero_arr2(3, 5, 7);
                testzero_arr1.set_zero();
                testzero_arr2.set_zero();
                BigPoly(5, 7, testzero_arr1.data(0)) = "Ax^3 + Bx^2";
                BigPoly(5, 7, testzero_arr1.data(1)) = "Cx^1";
                BigPoly(5, 7, testzero_arr1.data(2)) = "Dx^2 + Ex^1 + F";

                ntt_dot_product_bigpolyarray_nttbigpolyarray(testzero_arr1.data(0), testzero_arr2.data(0), 3, tables, result.data(), pool);
                Assert::IsTrue(result.to_string() == "0");

                // testing with BigPolyArray that should extract the ith entry of the other BigPolyArray
                BigPolyArray test_arr1(3, 5, 7);
                BigPolyArray test_arr2(3, 5, 7);
                test_arr1.set_zero();
                test_arr2.set_zero();
                BigPoly(5, 7, test_arr1.data(0)) = "6x^1 + 5";
                BigPoly(5, 7, test_arr1.data(1)) = "4x^3";
                BigPoly(5, 7, test_arr1.data(2)) = "3x^2 + 2x^1 + 1";
                BigPoly(5, 7, test_arr2.data(2)) = "1x^3 + 1x^2 + 1x^1 + 1";
                ntt_dot_product_bigpolyarray_nttbigpolyarray(test_arr1.data(0), test_arr2.data(0), 3, tables, result.data(), pool);
                Assert::IsTrue(result.to_string() == "3x^2 + 2x^1 + 1");

                // testing with BigPolys where a polymod reduction will occur
                BigPolyArray arr1(2, 5, 7);
                BigPolyArray arr2(2, 5, 7);
                BigPoly(5, 7, arr1.data(0)) = "1x^1";
                BigPoly(5, 7, arr1.data(1)) = "1x^3";
                BigPoly(5, 7, arr2.data(0)) = "1";
                BigPoly(5, 7, arr2.data(1)) = "2x^1";
                for (int i = 0; i < 2; i++)
                {
                    ntt_negacyclic_harvey(arr2.data(i), tables);
                }
                ntt_dot_product_bigpolyarray_nttbigpolyarray(arr1.data(0), arr2.data(0), 2, tables, result.data(), pool);
                Assert::IsTrue(result.to_string() == "1x^1 + 5F");

                // 1 BigPoly per array, each of which is a scalar, and mod reduction will occur
                BigPolyArray scalartest1(1, 5, 7);
                BigPolyArray scalartest2(1, 5, 7);
                BigPoly(5, 7, scalartest1.data(0)) = "2";
                BigPoly(5, 7, scalartest2.data(0)) = "4x^3 + 4x^2 + 4x^1 + 4";
                ntt_dot_product_bigpolyarray_nttbigpolyarray(scalartest1.data(0), scalartest2.data(0), 1, tables, result.data(), pool);
                Assert::IsTrue(result.to_string() == "8");

                // 1 BigPoly per array, each of which is a scalar, one of which is zero 
                BigPolyArray scalar_zero_test1(1, 5, 7);
                BigPolyArray scalar_zero_test2(1, 5, 7);
                BigPoly(5, 7, scalar_zero_test1.data(0)) = "17";
                BigPoly(5, 7, scalar_zero_test2.data(0)) = "0";
                ntt_dot_product_bigpolyarray_nttbigpolyarray(scalar_zero_test1.data(0), scalar_zero_test2.data(0), 1, tables, result.data(), pool);
                Assert::IsTrue(result.to_string() == "0");

                // General BigPolyArray, where we expect both mod and polymod reduction
                BigPolyArray general1(4, 5, 7);
                BigPolyArray general2(4, 5, 7);
                BigPoly(5, 7, general1.data(0)) = "3x^2 + 2x^1";
                BigPoly(5, 7, general1.data(1)) = "1x^1 + 5";
                BigPoly(5, 7, general1.data(2)) = "1x^2 + 27";
                BigPoly(5, 7, general1.data(3)) = "3x^2 + 1x^1";
                BigPoly(5, 7, general2.data(0)) = "1x^3";
                BigPoly(5, 7, general2.data(1)) = "2x^1 + 6";
                BigPoly(5, 7, general2.data(2)) = "3x^1 + A";
                BigPoly(5, 7, general2.data(3)) = "12x^2 + Bx^1";
                for (int i = 0; i < 4; i++)
                {
                    ntt_negacyclic_harvey(general2.data(i), tables);
                }
                ntt_dot_product_bigpolyarray_nttbigpolyarray(general1.data(0), general2.data(0), 4, tables, result.data(), pool);
                Assert::IsTrue(result.to_string() == "36x^3 + 17x^2 + 21x^1 + 49");

                // General BigPolyArray, where we expect both mod and polymod reduction
                BigPolyArray general3(2, 5, 7);
                BigPolyArray general4(2, 5, 7);
                BigPoly(5, 7, general3.data(0)) = "Ax^1 + 1";
                BigPoly(5, 7, general3.data(1)) = "Cx^2 + 3";
                BigPoly(5, 7, general4.data(0)) = "Bx^1 + 2";
                BigPoly(5, 7, general4.data(1)) = "Dx^3 + 4";
                for (int i = 0; i < 2; i++)
                {
                    ntt_negacyclic_harvey(general4.data(i), tables);
                }
                ntt_dot_product_bigpolyarray_nttbigpolyarray(general3.data(0), general4.data(0), 2, tables, result.data(), pool);
                Assert::IsTrue(result.to_string() == "27x^3 + 3Dx^2 + 45x^1 + E");
            }

            TEST_METHOD(NussbaumerMultiplyPolyPolyCoeffSmallMod)
            {
                MemoryPool &pool = *global_variables::global_memory_pool;
                BigPoly poly1(5, 64);
                BigPoly poly2(5, 64);
                BigPoly result(5, 64);
                poly1[0] = 5;
                poly1[1] = 1;
                poly1[2] = 3;
                poly1[3] = 2;
                poly2[0] = 7;
                poly2[1] = 7;
                poly2[3] = 2;
                SmallModulus mod(27);
                nussbaumer_multiply_poly_poly_coeffmod(poly1.data(), poly2.data(), 2, mod, result.data(), pool);
                Assert::IsTrue("12x^3 + 18x^2 + 9x^1 + 13" == result.to_string());

                const int coeff_power = 8;
                const int coeff_count = (1 << coeff_power) + 1;
                BigPoly poly3(coeff_count, 64);
                BigPoly poly4(coeff_count, 64);
                BigPoly polymod(coeff_count, 64);
                BigPoly correct(coeff_count, 64);
                BigPoly result2(coeff_count, 64);
                random_device rd;
                for (int i = 0; i < coeff_count - 1; i++)
                {
                    poly3[i] = rd() % mod.value();
                    poly4[i] = rd() % mod.value();
                }
                polymod[0] = 1;
                polymod[coeff_count - 1] = 1;
                PolyModulus polym(polymod.data(), coeff_count, 1);
                nonfft_multiply_poly_poly_polymod_coeffmod(poly3.data(), poly4.data(), polym, mod, correct.data(), pool);
                nussbaumer_multiply_poly_poly_coeffmod(poly3.data(), poly4.data(), coeff_power, mod, result2.data(), pool);
                for (int i = 0; i < coeff_count; ++i)
                {
                    Assert::IsTrue(correct[i] == result2[i]);
                }
            }
        };
    }
}