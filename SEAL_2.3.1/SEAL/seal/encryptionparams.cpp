#include "seal/encryptionparams.h"
#include "seal/chooser.h"
#include "seal/util/polycore.h"
#include "seal/util/uintarith.h"
#include "seal/util/modulus.h"
#include "seal/util/polymodulus.h"
#include "seal/defaultparams.h"
#include <stdexcept>
#include <limits>
#include <math.h>

using namespace std;
using namespace seal::util;

namespace seal
{
    EncryptionParameters::EncryptionParameters() : poly_modulus_(1, 1)
    {
        // It is important to ensure that poly_modulus always has at least one coefficient
        // and at least one uint64 per coefficient
        compute_hash();
    }

    void EncryptionParameters::save(ostream &stream) const
    {
        int32_t coeff_mod_count32 = static_cast<int>(coeff_modulus_.size());

        poly_modulus_.save(stream);
        stream.write(reinterpret_cast<const char*>(&coeff_mod_count32), sizeof(int32_t));
        for (int i = 0; i < coeff_mod_count32; i++)
        {
            coeff_modulus_[i].save(stream);
        }
        plain_modulus_.save(stream);
        stream.write(reinterpret_cast<const char*>(&noise_standard_deviation_), sizeof(double));
        stream.write(reinterpret_cast<const char*>(&noise_max_deviation_), sizeof(double));
    }

    void EncryptionParameters::load(istream &stream)
    {        
        poly_modulus_.load(stream);
        if (poly_modulus_.coeff_count() > SEAL_POLY_MOD_DEGREE_BOUND + 1 ||
            poly_modulus_.coeff_uint64_count() > 1)
        {
            throw std::invalid_argument("poly_modulus too large");
        }

        int32_t coeff_mod_count32 = 0;
        stream.read(reinterpret_cast<char*>(&coeff_mod_count32), sizeof(int32_t));
        if (coeff_mod_count32 > SEAL_COEFF_MOD_COUNT_BOUND || coeff_mod_count32 < 0)
        {
            throw std::invalid_argument("coeff_modulus too large");
        }
        coeff_modulus_.resize(coeff_mod_count32);
        for (int i = 0; i < coeff_mod_count32; i++)
        {
            coeff_modulus_[i].load(stream);
        }

        plain_modulus_.load(stream);

        stream.read(reinterpret_cast<char*>(&noise_standard_deviation_), sizeof(double));
        stream.read(reinterpret_cast<char*>(&noise_max_deviation_), sizeof(double));

        // Re-compute the hash
        compute_hash();
    }

    void EncryptionParameters::compute_hash()
    {
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());

        int total_uint64_count = 
            poly_modulus_.coeff_uint64_count() * poly_modulus_.coeff_count() +
            coeff_mod_count + 
            plain_modulus_.uint64_count() +
            1 + // noise_standard_deviation
            1; // noise_max_deviation

        Pointer param_data(allocate_uint(total_uint64_count, MemoryPoolHandle::Global()));
        uint64_t *param_data_ptr = param_data.get();

        set_poly_poly(poly_modulus_.data(), poly_modulus_.coeff_count(), 
            poly_modulus_.coeff_uint64_count(), param_data_ptr);
        param_data_ptr += poly_modulus_.coeff_uint64_count() * poly_modulus_.coeff_count();

        for (int i = 0; i < coeff_mod_count; i++)
        {
            *param_data_ptr = coeff_modulus_[i].value();
            param_data_ptr++;
        }

        set_uint_uint(plain_modulus_.data(), plain_modulus_.uint64_count(), param_data_ptr);
        param_data_ptr += plain_modulus_.uint64_count();

        memcpy(param_data_ptr++, &noise_standard_deviation_, sizeof(double));
        memcpy(param_data_ptr, &noise_max_deviation_, sizeof(double));

        HashFunction::sha3_hash(param_data.get(), total_uint64_count, hash_block_);
    }
}
