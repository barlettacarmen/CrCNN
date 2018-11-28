using Microsoft.VisualStudio.TestTools.UnitTesting;
using Microsoft.Research.SEAL;
using System.Collections.Generic;
using System;

namespace SEALNETTest
{
    [TestClass]
    public class KeyGeneratorWrapper
    {
        [TestMethod]
        public void BFVKeyGenerationNET()
        {
            var parms = new EncryptionParameters();
            {
                parms.NoiseStandardDeviation = 3.19;
                parms.PolyModulus = "1x^64 + 1";
                parms.CoeffModulus = new List<SmallModulus> { DefaultParams.SmallMods60Bit(0) };
                parms.PlainModulus = 1 << 6;

                var context = new SEALContext(parms);
                var keygen = new KeyGenerator(context);

                Assert.IsTrue(keygen.PublicKey.HashBlock.Equals(parms.HashBlock));
                Assert.IsTrue(keygen.SecretKey.HashBlock.Equals(parms.HashBlock));

                var evk = new EvaluationKeys();
                keygen.GenerateEvaluationKeys(60, evk);
                Assert.AreEqual(evk.HashBlock, parms.HashBlock);
                Assert.AreEqual(2, evk.Key(2)[0].Size);

                keygen.GenerateEvaluationKeys(30, 1, evk);
                Assert.AreEqual(evk.HashBlock, parms.HashBlock);
                Assert.AreEqual(4, evk.Key(2)[0].Size);

                keygen.GenerateEvaluationKeys(2, 2, evk);
                Assert.AreEqual(evk.HashBlock, parms.HashBlock);
                Assert.AreEqual(60, evk.Key(2)[0].Size);

                var galks = new GaloisKeys();
                keygen.GenerateGaloisKeys(60, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.AreEqual(2, galks.Key(3)[0].Size);
                Assert.AreEqual(10, galks.Size);

                keygen.GenerateGaloisKeys(30, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.AreEqual(4, galks.Key(3)[0].Size);
                Assert.AreEqual(10, galks.Size);

                keygen.GenerateGaloisKeys(2, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.AreEqual(60, galks.Key(3)[0].Size);
                Assert.AreEqual(10, galks.Size);

                keygen.GenerateGaloisKeys(60, new List<UInt64> { 1, 3, 5, 7 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(3));
                Assert.IsTrue(galks.HasKey(5));
                Assert.IsTrue(galks.HasKey(7));
                Assert.IsFalse(galks.HasKey(9));
                Assert.IsFalse(galks.HasKey(127));
                Assert.AreEqual(2, galks.Key(1)[0].Size);
                Assert.AreEqual(2, galks.Key(3)[0].Size);
                Assert.AreEqual(2, galks.Key(5)[0].Size);
                Assert.AreEqual(2, galks.Key(7)[0].Size);
                Assert.AreEqual(4, galks.Size);

                keygen.GenerateGaloisKeys(30, new List<UInt64> { 1, 3, 5, 7 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(3));
                Assert.IsTrue(galks.HasKey(5));
                Assert.IsTrue(galks.HasKey(7));
                Assert.IsFalse(galks.HasKey(9));
                Assert.IsFalse(galks.HasKey(127));
                Assert.AreEqual(4, galks.Key(1)[0].Size);
                Assert.AreEqual(4, galks.Key(3)[0].Size);
                Assert.AreEqual(4, galks.Key(5)[0].Size);
                Assert.AreEqual(4, galks.Key(7)[0].Size);
                Assert.AreEqual(4, galks.Size);

                keygen.GenerateGaloisKeys(2, new List<UInt64> { 1, 3, 5, 7 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(3));
                Assert.IsTrue(galks.HasKey(5));
                Assert.IsTrue(galks.HasKey(7));
                Assert.IsFalse(galks.HasKey(9));
                Assert.IsFalse(galks.HasKey(127));
                Assert.AreEqual(60, galks.Key(1)[0].Size);
                Assert.AreEqual(60, galks.Key(3)[0].Size);
                Assert.AreEqual(60, galks.Key(5)[0].Size);
                Assert.AreEqual(60, galks.Key(7)[0].Size);
                Assert.AreEqual(4, galks.Size);

                keygen.GenerateGaloisKeys(30, new List<UInt64> { 1 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsFalse(galks.HasKey(3));
                Assert.IsFalse(galks.HasKey(127));
                Assert.AreEqual(4, galks.Key(1)[0].Size);
                Assert.AreEqual(1, galks.Size);

                keygen.GenerateGaloisKeys(30, new List<UInt64> { 127 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsFalse(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(127));
                Assert.AreEqual(4, galks.Key(127)[0].Size);
                Assert.AreEqual(1, galks.Size);
            }
            {
                parms.NoiseStandardDeviation = 3.19;
                parms.PolyModulus = "1x^256 + 1";
                parms.CoeffModulus = new List<SmallModulus> {
                    DefaultParams.SmallMods60Bit(0), DefaultParams.SmallMods30Bit(0), DefaultParams.SmallMods30Bit(1) };
                parms.PlainModulus = 1 << 6;

                var context = new SEALContext(parms);
                var keygen = new KeyGenerator(context);

                Assert.AreEqual(keygen.PublicKey.HashBlock, parms.HashBlock);
                Assert.AreEqual(keygen.SecretKey.HashBlock, parms.HashBlock);

                var evk = new EvaluationKeys();
                keygen.GenerateEvaluationKeys(60, 2, evk);
                Assert.AreEqual(evk.HashBlock, parms.HashBlock);
                Assert.AreEqual(2, evk.Key(2)[0].Size);

                keygen.GenerateEvaluationKeys(30, 2, evk);
                Assert.AreEqual(evk.HashBlock, parms.HashBlock);
                Assert.AreEqual(4, evk.Key(2)[0].Size);

                keygen.GenerateEvaluationKeys(4, 1, evk);
                Assert.AreEqual(evk.HashBlock, parms.HashBlock);
                Assert.AreEqual(30, evk.Key(2)[0].Size);

                var galks = new GaloisKeys();
                keygen.GenerateGaloisKeys(60, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.AreEqual(2, galks.Key(3)[0].Size);
                Assert.AreEqual(14, galks.Size);

                keygen.GenerateGaloisKeys(30, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.AreEqual(4, galks.Key(3)[0].Size);
                Assert.AreEqual(14, galks.Size);

                keygen.GenerateGaloisKeys(2, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.AreEqual(60, galks.Key(3)[0].Size);
                Assert.AreEqual(14, galks.Size);

                keygen.GenerateGaloisKeys(60, new List<UInt64> { 1, 3, 5, 7 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(3));
                Assert.IsTrue(galks.HasKey(5));
                Assert.IsTrue(galks.HasKey(7));
                Assert.IsFalse(galks.HasKey(9));
                Assert.IsFalse(galks.HasKey(511));
                Assert.AreEqual(2, galks.Key(1)[0].Size);
                Assert.AreEqual(2, galks.Key(3)[0].Size);
                Assert.AreEqual(2, galks.Key(5)[0].Size);
                Assert.AreEqual(2, galks.Key(7)[0].Size);
                Assert.AreEqual(4, galks.Size);

                keygen.GenerateGaloisKeys(30, new List<UInt64> { 1, 3, 5, 7 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(3));
                Assert.IsTrue(galks.HasKey(5));
                Assert.IsTrue(galks.HasKey(7));
                Assert.IsFalse(galks.HasKey(9));
                Assert.IsFalse(galks.HasKey(511));
                Assert.AreEqual(4, galks.Key(1)[0].Size);
                Assert.AreEqual(4, galks.Key(3)[0].Size);
                Assert.AreEqual(4, galks.Key(5)[0].Size);
                Assert.AreEqual(4, galks.Key(7)[0].Size);
                Assert.AreEqual(4, galks.Size);

                keygen.GenerateGaloisKeys(2, new List<UInt64> { 1, 3, 5, 7 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(3));
                Assert.IsTrue(galks.HasKey(5));
                Assert.IsTrue(galks.HasKey(7));
                Assert.IsFalse(galks.HasKey(9));
                Assert.IsFalse(galks.HasKey(511));
                Assert.AreEqual(60, galks.Key(1)[0].Size);
                Assert.AreEqual(60, galks.Key(3)[0].Size);
                Assert.AreEqual(60, galks.Key(5)[0].Size);
                Assert.AreEqual(60, galks.Key(7)[0].Size);
                Assert.AreEqual(4, galks.Size);

                keygen.GenerateGaloisKeys(30, new List<UInt64> { 1 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsTrue(galks.HasKey(1));
                Assert.IsFalse(galks.HasKey(3));
                Assert.IsFalse(galks.HasKey(511));
                Assert.AreEqual(4, galks.Key(1)[0].Size);
                Assert.AreEqual(1, galks.Size);

                keygen.GenerateGaloisKeys(30, new List<UInt64> { 511 }, galks);
                Assert.AreEqual(galks.HashBlock, parms.HashBlock);
                Assert.IsFalse(galks.HasKey(1));
                Assert.IsTrue(galks.HasKey(511));
                Assert.AreEqual(4, galks.Key(511)[0].Size);
                Assert.AreEqual(1, galks.Size);
            }
        }
    }
}