#include "CppUnitTest.h"
#include "seal/context.h"
#include "seal/keygenerator.h"
#include "seal/util/polycore.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace seal;
using namespace seal::util;
using namespace std;

namespace SEALTest
{
    TEST_CLASS(KeyGeneratorTest)
    {
    public:
        TEST_METHOD(FVKeyGeneration)
        {
            EncryptionParameters parms;
            {
                parms.set_noise_standard_deviation(3.19);
                parms.set_poly_modulus("1x^64 + 1");
                parms.set_plain_modulus(1 << 6);
                parms.set_coeff_modulus({ small_mods_60bit(0) });
                SEALContext context(parms);
                KeyGenerator keygen(context);
                Assert::IsFalse(keygen.public_key().data().is_zero());
                Assert::IsFalse(keygen.secret_key().data().is_zero());

                EvaluationKeys evk;
                keygen.generate_evaluation_keys(60, evk);
                Assert::IsTrue(evk.hash_block() == parms.hash_block());
                Assert::AreEqual(2, evk.key(2)[0].size());
                for (int j = 0; j < evk.size(); j++)
                {
                    for (int i = 0; i < evk.key(j + 2).size(); i++)
                    {
                        for (int k = 0; k < evk.key(j + 2)[i].size(); k++)
                        {
                            Assert::IsFalse(is_zero_poly(evk.key(j + 2)[i].data(k), evk.key(j + 2)[i].poly_coeff_count(), evk.key(j + 2)[i].coeff_mod_count()));
                        }
                    }
                }

                keygen.generate_evaluation_keys(30, 1, evk);
                Assert::IsTrue(evk.hash_block() == parms.hash_block());
                Assert::AreEqual(4, evk.key(2)[0].size());
                for (int j = 0; j < evk.size(); j++)
                {
                    for (int i = 0; i < evk.key(j + 2).size(); i++)
                    {
                        for (int k = 0; k < evk.key(j + 2)[i].size(); k++)
                        {
                            Assert::IsFalse(is_zero_poly(evk.key(j + 2)[i].data(k), evk.key(j + 2)[i].poly_coeff_count(), evk.key(j + 2)[i].coeff_mod_count()));
                        }
                    }
                }

                keygen.generate_evaluation_keys(2, 2, evk);
                Assert::IsTrue(evk.hash_block() == parms.hash_block());
                Assert::AreEqual(60, evk.key(2)[0].size());
                for (int j = 0; j < evk.size(); j++)
                {
                    for (int i = 0; i < evk.key(j + 2).size(); i++)
                    {
                        for (int k = 0; k < evk.key(j + 2)[i].size(); k++)
                        {
                            Assert::IsFalse(is_zero_poly(evk.key(j + 2)[i].data(k), evk.key(j + 2)[i].poly_coeff_count(), evk.key(j + 2)[i].coeff_mod_count()));
                        }
                    }
                }

                GaloisKeys galks;
                keygen.generate_galois_keys(60, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::AreEqual(2, galks.key(3)[0].size());
                Assert::AreEqual(10, galks.size());

                keygen.generate_galois_keys(30, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::AreEqual(4, galks.key(3)[0].size());
                Assert::AreEqual(10, galks.size());

                keygen.generate_galois_keys(2, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::AreEqual(60, galks.key(3)[0].size());
                Assert::AreEqual(10, galks.size());

                keygen.generate_galois_keys(60, { 1, 3, 5, 7 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsTrue(galks.has_key(3));
                Assert::IsTrue(galks.has_key(5));
                Assert::IsTrue(galks.has_key(7));
                Assert::IsFalse(galks.has_key(9));
                Assert::IsFalse(galks.has_key(127));
                Assert::AreEqual(2, galks.key(1)[0].size());
                Assert::AreEqual(2, galks.key(3)[0].size());
                Assert::AreEqual(2, galks.key(5)[0].size());
                Assert::AreEqual(2, galks.key(7)[0].size());
                Assert::AreEqual(4, galks.size());

                keygen.generate_galois_keys(30, { 1, 3, 5, 7 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsTrue(galks.has_key(3));
                Assert::IsTrue(galks.has_key(5));
                Assert::IsTrue(galks.has_key(7));
                Assert::IsFalse(galks.has_key(9));
                Assert::IsFalse(galks.has_key(127));
                Assert::AreEqual(4, galks.key(1)[0].size());
                Assert::AreEqual(4, galks.key(3)[0].size());
                Assert::AreEqual(4, galks.key(5)[0].size());
                Assert::AreEqual(4, galks.key(7)[0].size());
                Assert::AreEqual(4, galks.size());

                keygen.generate_galois_keys(2, { 1, 3, 5, 7 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsTrue(galks.has_key(3));
                Assert::IsTrue(galks.has_key(5));
                Assert::IsTrue(galks.has_key(7));
                Assert::IsFalse(galks.has_key(9));
                Assert::IsFalse(galks.has_key(127));
                Assert::AreEqual(60, galks.key(1)[0].size());
                Assert::AreEqual(60, galks.key(3)[0].size());
                Assert::AreEqual(60, galks.key(5)[0].size());
                Assert::AreEqual(60, galks.key(7)[0].size());
                Assert::AreEqual(4, galks.size());

                keygen.generate_galois_keys(30, { 1 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsFalse(galks.has_key(3));
                Assert::IsFalse(galks.has_key(127));
                Assert::AreEqual(4, galks.key(1)[0].size());
                Assert::AreEqual(1, galks.size());

                keygen.generate_galois_keys(30, { 127 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsFalse(galks.has_key(1));
                Assert::IsTrue(galks.has_key(127));
                Assert::AreEqual(4, galks.key(127)[0].size());
                Assert::AreEqual(1, galks.size());
            }
            {
                parms.set_noise_standard_deviation(3.19);
                parms.set_poly_modulus("1x^256 + 1");
                parms.set_plain_modulus(1 << 6);
                parms.set_coeff_modulus({ small_mods_60bit(0), small_mods_30bit(0), small_mods_30bit(1) });
                SEALContext context(parms);
                KeyGenerator keygen(context);
                Assert::IsFalse(keygen.public_key().data().is_zero());
                Assert::IsFalse(keygen.secret_key().data().is_zero());

                EvaluationKeys evk;
                keygen.generate_evaluation_keys(60, 2, evk);
                Assert::IsTrue(evk.hash_block() == parms.hash_block());
                Assert::AreEqual(2, evk.key(2)[0].size());
                for (int j = 0; j < evk.size(); j++)
                {
                    for (int i = 0; i < evk.key(j + 2).size(); i++)
                    {
                        for (int k = 0; k < evk.key(j + 2)[i].size(); k++)
                        {
                            Assert::IsFalse(is_zero_poly(evk.key(j + 2)[i].data(k), evk.key(j + 2)[i].poly_coeff_count(), evk.key(j + 2)[i].coeff_mod_count()));
                        }
                    }
                }

                keygen.generate_evaluation_keys(30, 2, evk);
                Assert::IsTrue(evk.hash_block() == parms.hash_block());
                Assert::AreEqual(4, evk.key(2)[0].size());
                for (int j = 0; j < evk.size(); j++)
                {
                    for (int i = 0; i < evk.key(j + 2).size(); i++)
                    {
                        for (int k = 0; k < evk.key(j + 2)[i].size(); k++)
                        {
                            Assert::IsFalse(is_zero_poly(evk.key(j + 2)[i].data(k), evk.key(j + 2)[i].poly_coeff_count(), evk.key(j + 2)[i].coeff_mod_count()));
                        }
                    }
                }

                keygen.generate_evaluation_keys(4, 1, evk);
                Assert::IsTrue(evk.hash_block() == parms.hash_block());
                Assert::AreEqual(30, evk.key(2)[0].size());
                for (int j = 0; j < evk.size(); j++)
                {
                    for (int i = 0; i < evk.key(j + 2).size(); i++)
                    {
                        for (int k = 0; k < evk.key(j + 2)[i].size(); k++)
                        {
                            Assert::IsFalse(is_zero_poly(evk.key(j + 2)[i].data(k), evk.key(j + 2)[i].poly_coeff_count(), evk.key(j + 2)[i].coeff_mod_count()));
                        }
                    }
                }

                GaloisKeys galks;
                keygen.generate_galois_keys(60, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::AreEqual(2, galks.key(3)[0].size());
                Assert::AreEqual(14, galks.size());

                keygen.generate_galois_keys(30, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::AreEqual(4, galks.key(3)[0].size());
                Assert::AreEqual(14, galks.size());

                keygen.generate_galois_keys(2, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::AreEqual(60, galks.key(3)[0].size());
                Assert::AreEqual(14, galks.size());

                keygen.generate_galois_keys(60, { 1, 3, 5, 7 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsTrue(galks.has_key(3));
                Assert::IsTrue(galks.has_key(5));
                Assert::IsTrue(galks.has_key(7));
                Assert::IsFalse(galks.has_key(9));
                Assert::IsFalse(galks.has_key(511));
                Assert::AreEqual(2, galks.key(1)[0].size());
                Assert::AreEqual(2, galks.key(3)[0].size());
                Assert::AreEqual(2, galks.key(5)[0].size());
                Assert::AreEqual(2, galks.key(7)[0].size());
                Assert::AreEqual(4, galks.size());

                keygen.generate_galois_keys(30, { 1, 3, 5, 7 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsTrue(galks.has_key(3));
                Assert::IsTrue(galks.has_key(5));
                Assert::IsTrue(galks.has_key(7));
                Assert::IsFalse(galks.has_key(9));
                Assert::IsFalse(galks.has_key(511));
                Assert::AreEqual(4, galks.key(1)[0].size());
                Assert::AreEqual(4, galks.key(3)[0].size());
                Assert::AreEqual(4, galks.key(5)[0].size());
                Assert::AreEqual(4, galks.key(7)[0].size());
                Assert::AreEqual(4, galks.size());

                keygen.generate_galois_keys(2, { 1, 3, 5, 7 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsTrue(galks.has_key(3));
                Assert::IsTrue(galks.has_key(5));
                Assert::IsTrue(galks.has_key(7));
                Assert::IsFalse(galks.has_key(9));
                Assert::IsFalse(galks.has_key(511));
                Assert::AreEqual(60, galks.key(1)[0].size());
                Assert::AreEqual(60, galks.key(3)[0].size());
                Assert::AreEqual(60, galks.key(5)[0].size());
                Assert::AreEqual(60, galks.key(7)[0].size());
                Assert::AreEqual(4, galks.size());

                keygen.generate_galois_keys(30, { 1 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsTrue(galks.has_key(1));
                Assert::IsFalse(galks.has_key(3));
                Assert::IsFalse(galks.has_key(511));
                Assert::AreEqual(4, galks.key(1)[0].size());
                Assert::AreEqual(1, galks.size());

                keygen.generate_galois_keys(30, { 511 }, galks);
                Assert::IsTrue(galks.hash_block() == parms.hash_block());
                Assert::IsFalse(galks.has_key(1));
                Assert::IsTrue(galks.has_key(511));
                Assert::AreEqual(4, galks.key(511)[0].size());
                Assert::AreEqual(1, galks.size());
            }
        }
    };
}
