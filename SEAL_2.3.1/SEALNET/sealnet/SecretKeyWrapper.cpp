#include "sealnet/SecretKeyWrapper.h"
#include "sealnet/Common.h"

using namespace std;
using namespace System;
using namespace System::IO;
using namespace System::Collections::Generic;

namespace Microsoft
{
    namespace Research
    {
        namespace SEAL
        {
            SecretKey::SecretKey()
            {
                try
                {
                    sk_ = new seal::SecretKey();
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            SecretKey::SecretKey(SecretKey ^copy)
            {
                if (copy == nullptr)
                {
                    throw gcnew ArgumentNullException("copy cannot be null");
                }
                try
                {
                    sk_ = new seal::SecretKey(copy->GetKey());
                    GC::KeepAlive(copy);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            void SecretKey::Set(SecretKey ^assign)
            {
                if (sk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SecretKey is disposed");
                }
                if (assign == nullptr)
                {
                    throw gcnew ArgumentNullException("assign cannot be null");
                }
                try
                {
                    *sk_ = *assign->sk_;
                    GC::KeepAlive(assign);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            BigPoly ^SecretKey::Data::get()
            {
                if (sk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SecretKey is disposed");
                }
                return gcnew BigPoly(sk_->data());
            }

            void SecretKey::Save(Stream ^stream)
            {
                if (sk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SecretKey is disposed");
                }
                if (stream == nullptr)
                {
                    throw gcnew ArgumentNullException("stream cannot be null");
                }
                try
                {
                    Write(stream, reinterpret_cast<const char*>(&sk_->hash_block()), 
                        sizeof(seal::EncryptionParameters::hash_block_type));
                    auto poly = gcnew BigPoly(&sk_->data());
                    poly->Save(stream);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            void SecretKey::Load(Stream ^stream)
            {
                if (sk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SecretKey is disposed");
                }
                if (stream == nullptr)
                {
                    throw gcnew ArgumentNullException("stream cannot be null");
                }
                try
                {
                    Read(stream, reinterpret_cast<char*>(&sk_->hash_block()), 
                        sizeof(seal::EncryptionParameters::hash_block_type));
                    auto poly = gcnew BigPoly(&sk_->data());
                    poly->Load(stream);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            SecretKey::~SecretKey()
            {
                this->!SecretKey();
            }

            SecretKey::!SecretKey()
            {
                if (sk_ != nullptr)
                {
                    delete sk_;
                    sk_ = nullptr;
                }
            }

            Tuple<UInt64, UInt64, UInt64, UInt64> ^SecretKey::HashBlock::get()
            {
                if (sk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SecretKey is disposed");
                }
                return gcnew Tuple<UInt64, UInt64, UInt64, UInt64>(
                    sk_->hash_block()[0],
                    sk_->hash_block()[1],
                    sk_->hash_block()[2],
                    sk_->hash_block()[3]);
            }

            SecretKey::SecretKey(const seal::SecretKey &ciphertext) : sk_(nullptr)
            {
                try
                {
                    sk_ = new seal::SecretKey(ciphertext);
                }
                catch (const exception &e)
                {
                    HandleException(&e);
                }
                catch (...)
                {
                    HandleException(nullptr);
                }
            }

            seal::SecretKey &SecretKey::GetKey()
            {
                if (sk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("SecretKey is disposed");
                }
                return *sk_;
            }
        }
    }
}