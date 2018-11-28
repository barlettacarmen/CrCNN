#include "sealnet/PublicKeyWrapper.h"
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
            PublicKey::PublicKey()
            {
                try
                {
                    pk_ = new seal::PublicKey();
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

            PublicKey::PublicKey(PublicKey ^copy)
            {
                if (copy == nullptr)
                {
                    throw gcnew ArgumentNullException("copy cannot be null");
                }
                try
                {
                    pk_ = new seal::PublicKey(copy->GetKey());
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

            void PublicKey::Set(PublicKey ^assign)
            {
                if (pk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("PublicKey is disposed");
                }
                if (assign == nullptr)
                {
                    throw gcnew ArgumentNullException("assign cannot be null");
                }
                try
                {
                    *pk_ = *assign->pk_;
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

            BigPolyArray ^PublicKey::Data::get()
            {
                if (pk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("PublicKey is disposed");
                }
                return gcnew BigPolyArray(pk_->data());
            }

            void PublicKey::Save(Stream ^stream)
            {
                if (pk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("PublicKey is disposed");
                }
                if (stream == nullptr)
                {
                    throw gcnew ArgumentNullException("stream cannot be null");
                }
                try
                {
                    Write(stream, reinterpret_cast<const char*>(&pk_->hash_block()), 
                        sizeof(seal::EncryptionParameters::hash_block_type));
                    auto poly_array = gcnew BigPolyArray(&pk_->data());
                    poly_array->Save(stream);
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

            void PublicKey::Load(Stream ^stream)
            {
                if (pk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("PublicKey is disposed");
                }
                if (stream == nullptr)
                {
                    throw gcnew ArgumentNullException("stream cannot be null");
                }
                try
                {
                    Read(stream, reinterpret_cast<char*>(&pk_->hash_block()), 
                        sizeof(seal::EncryptionParameters::hash_block_type));
                    auto poly_array = gcnew BigPolyArray(&pk_->data());
                    poly_array->Load(stream);
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

            Tuple<UInt64, UInt64, UInt64, UInt64> ^PublicKey::HashBlock::get()
            {
                if (pk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("PublicKey is disposed");
                }
                return gcnew Tuple<UInt64, UInt64, UInt64, UInt64>(
                    pk_->hash_block()[0],
                    pk_->hash_block()[1],
                    pk_->hash_block()[2],
                    pk_->hash_block()[3]);
            }

            PublicKey::~PublicKey()
            {
                this->!PublicKey();
            }

            PublicKey::!PublicKey()
            {
                if (pk_ != nullptr)
                {
                    delete pk_;
                    pk_ = nullptr;
                }
            }

            PublicKey::PublicKey(const seal::PublicKey &ciphertext) : pk_(nullptr)
            {
                try
                {
                    pk_ = new seal::PublicKey(ciphertext);
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

            seal::PublicKey &PublicKey::GetKey()
            {
                if (pk_ == nullptr)
                {
                    throw gcnew ObjectDisposedException("PublicKey is disposed");
                }
                return *pk_;
            }
        }
    }
}