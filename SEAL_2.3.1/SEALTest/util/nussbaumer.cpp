#include "CppUnitTest.h"
#include "seal/util/nussbaumer.h"
#include "seal/util/polyarith.h"
#include "seal/util/uintarith.h"
#include "seal/bigpoly.h"
#include <random>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace seal;
using namespace seal::util;
using namespace std;

namespace SEALTest
{
    namespace util
    {
        TEST_CLASS(PolyFFTMult)
        {
        public:
            TEST_METHOD(NussbaumerMultiplyPolyPolyBase)
            {
                BigPoly poly1(5, 128);
                BigPoly poly2(5, 128);
                BigPoly result(5, 128);
                poly1[0] = 5;
                poly1[1] = 1;
                poly1[2] = 3;
                poly1[3] = 8;
                poly2[0] = 7;
                poly2[1] = 7;
                poly2[3] = 2;
                MemoryPool &pool = *global_variables::global_memory_pool;
                nussbaumer_multiply_poly_poly(poly1.data(), poly2.data(), 2, 2, 2, 2, result.data(), pool);
                Assert::IsTrue("57x^3 + Cx^2 + 24x^1 + FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE9" == result.to_string());

                result.resize(5, 64);
                result.set_zero();
                nussbaumer_multiply_poly_poly(poly1.data(), poly2.data(), 2, 2, 2, 1, result.data(), pool);
                Assert::IsTrue("57x^3 + Cx^2 + 24x^1 + FFFFFFFFFFFFFFE9" == result.to_string());

                result.resize(5, 128);
                result.set_zero();
                poly1.resize(5, 64);
                poly2.resize(5, 64);
                nussbaumer_multiply_poly_poly(poly1.data(), poly2.data(), 2, 1, 2,2, result.data(), pool);
                Assert::IsTrue("57x^3 + Cx^2 + 24x^1 + FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFE9" == result.to_string());
            }

            TEST_METHOD(NussbaumerMultiplyPolyPoly)
            {
                const int coeff_power = 8;
                const int coeff_count = (1 << coeff_power) + 1;
                BigPoly poly1(coeff_count, 128);
                BigPoly poly2(coeff_count, 128);
                BigPoly polymod(coeff_count, 128);
                BigPoly correct(2 * coeff_count, 128);
                BigPoly result(coeff_count, 128);
                default_random_engine random;
                random.seed(0);
                for (int i = 0; i < coeff_count - 1; ++i)
                {
                    poly1[i] = random() % 100;
                    poly2[i] = random() % 100;
                }
                polymod[0] = 1;
                polymod[coeff_count - 1] = 1;
                MemoryPool &pool = *global_variables::global_memory_pool;
                multiply_poly_poly(poly1.data(), coeff_count, 2, poly2.data(), coeff_count, 2, 2 * coeff_count, 2, correct.data(), pool);
                for (int i = coeff_count - 1; i < 2 * coeff_count; ++i)
                {
                    uint64_t *lower_coeff = get_poly_coeff(correct.data(), i - (coeff_count - 1), 2);
                    uint64_t *upper_coeff = get_poly_coeff(correct.data(), i, 2);
                    sub_uint_uint(lower_coeff, upper_coeff, 2, lower_coeff);
                    set_zero_uint(2, upper_coeff);
                }
                nussbaumer_multiply_poly_poly(poly1.data(), poly2.data(), coeff_power, 2, 2, 2, result.data(), pool);
                for (int i = 0; i < coeff_count; ++i)
                {
                    Assert::IsTrue(correct[i] == result[i]);
                }
            }

            TEST_METHOD(NussbaumerCrossMultiplyPolyPolyBase)
            {
                BigPoly poly1(5, 64);
                BigPoly poly2(5, 64);
                BigPoly result11(5, 128);
                BigPoly result12(5, 128);
                BigPoly result22(5, 128);

                poly1[0] = 5;
                poly1[1] = 1;
                poly1[2] = 3;
                poly1[3] = 1;

                poly2[0] = 7;
                poly2[1] = 7;
                poly2[2] = 3;
                poly2[3] = 2;

                int coeff_count_power = 2; 

                MemoryPool &pool = *global_variables::global_memory_pool;
                nussbaumer_cross_multiply_poly_poly(poly1.data(), poly2.data(), coeff_count_power, 1, 1, 2, result11.data(), result22.data(), result12.data(),pool);

                BigPoly result11_correct(5, 128); 
                nussbaumer_multiply_poly_poly(poly1.data(), poly1.data(), coeff_count_power, 1, 1, 2, result11_correct.data(), pool);
                Assert::AreEqual(result11_correct.data()[0], result11.data()[0]);
                Assert::AreEqual(result11.data()[1], result11_correct.data()[1]);
                Assert::AreEqual(result11.data()[2], result11_correct.data()[2]);
                Assert::AreEqual(result11.data()[3], result11_correct.data()[3]);

                BigPoly result22_correct(5, 128);
                nussbaumer_multiply_poly_poly(poly2.data(), poly2.data(), coeff_count_power, 1, 1, 2, result22_correct.data(), pool);
                for (int i = 0; i < 4; i++)
                {
                    Assert::AreEqual(result22_correct.data()[i], result22.data()[i]);
                }

                BigPoly result12_correct(5, 128);
                nussbaumer_multiply_poly_poly(poly1.data(), poly2.data(), coeff_count_power, 1, 1, 2, result12_correct.data(), pool);
                for (int i = 0; i < 4; i++)
                {
                    Assert::AreEqual(result12_correct.data()[i], result12.data()[i]);
                }

            }

            TEST_METHOD(NussbaumerCrossMultiplyPolyPoly)
            {
                const int coeff_power = 8;
                const int coeff_count = (1 << coeff_power) + 1;
                BigPoly poly1(coeff_count, 128);
                BigPoly poly2(coeff_count, 128);
                BigPoly polymod(coeff_count, 128);
                BigPoly correct11(coeff_count, 128);
                BigPoly correct12(coeff_count, 128);
                BigPoly correct22(coeff_count, 128);
                BigPoly result11(coeff_count, 128);
                BigPoly result12(coeff_count, 128);
                BigPoly result22(coeff_count, 128);

                default_random_engine random;
                random.seed(0);
                for (int i = 0; i < coeff_count - 1; ++i)
                {
                    poly1[i] = random() % 100;
                    poly2[i] = random() % 100;
                }
                polymod[0] = 1;
                polymod[coeff_count - 1] = 1;
                MemoryPool &pool = *global_variables::global_memory_pool;

                nussbaumer_cross_multiply_poly_poly(poly1.data(), poly2.data(), coeff_power, 1, 2, 2, result11.data(), result22.data(), result12.data(), pool);


                nussbaumer_multiply_poly_poly(poly1.data(), poly1.data(), coeff_power, 1, 2, 2, correct11.data(), pool);
                nussbaumer_multiply_poly_poly(poly1.data(), poly2.data(), coeff_power, 1, 2, 2, correct12.data(), pool);
                nussbaumer_multiply_poly_poly(poly2.data(), poly2.data(), coeff_power, 1, 2, 2, correct22.data(), pool);

                for (int i = 0; i < coeff_count - 1; ++i)
                {
                    Assert::IsTrue(correct11[i] == result11[i]);
                    Assert::IsTrue(correct12[i] == result12[i]);
                    Assert::IsTrue(correct22[i] == result22[i]);
                }
            }
        };
    }
}