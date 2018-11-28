#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "seal/evaluator.h"
#include "seal/util/common.h"
#include "seal/util/uintcore.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/polycore.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/polyfftmultsmallmod.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    Evaluator::Evaluator(const SEALContext &context) :
        parms_(context.parms()), qualifiers_(context.qualifiers()), 
        base_converter_(context.base_converter_), 
        coeff_modulus_(context.coeff_modulus()) 
    {
        // Verify parameters
        if (!qualifiers_.parameters_set)
        {
            throw invalid_argument("encryption parameters are not set correctly");
        }
        
        int coeff_count = parms_.poly_modulus().coeff_count();
        int poly_coeff_uint64_count = parms_.poly_modulus().coeff_uint64_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        bsk_base_mod_count_ = base_converter_.bsk_base_mod_count();
        
        // Set SmallNTTTables
        bsk_small_ntt_tables_.resize(bsk_base_mod_count_);
        bsk_small_ntt_tables_ = base_converter_.get_bsk_small_ntt_table();

        coeff_small_ntt_tables_.resize(coeff_mod_count);
        coeff_small_ntt_tables_ = context.small_ntt_tables_;

        // Copy over bsk moduli array
        bsk_mod_array_ = base_converter_.get_bsk_mod_array();

        // Copy over inverse of coeff moduli products mod each coeff moduli
        inv_coeff_products_mod_coeff_array_ = base_converter_.get_inv_coeff_mod_coeff_array();

        // Populate coeff products array for compose functions (used in noise budget)
        coeff_products_array_ = allocate_uint(coeff_mod_count * coeff_mod_count, pool_);
        Pointer tmp_coeff(allocate_uint(coeff_mod_count, pool_));
        set_zero_uint(coeff_mod_count * coeff_mod_count, coeff_products_array_.get());

        for (int i = 0; i < coeff_mod_count; i++)
        {
            *(coeff_products_array_.get() + (i * coeff_mod_count)) = 1;
            for (int j = 0; j < coeff_mod_count; j++)
            {
                if (i != j)
                {
                    multiply_uint_uint64(coeff_products_array_.get() + (i * coeff_mod_count), coeff_mod_count, coeff_modulus_[j].value(), coeff_mod_count, tmp_coeff.get());
                    set_uint_uint(tmp_coeff.get(), coeff_mod_count, coeff_products_array_.get() + (i * coeff_mod_count));
                }
            }
        }

        // Calculate coeff_modulus / plain_modulus.
        coeff_div_plain_modulus_ = allocate_uint(coeff_mod_count, pool_);
        ConstPointer wide_plain_modulus(duplicate_uint_if_needed(parms_.plain_modulus().data(), parms_.plain_modulus().uint64_count(), coeff_mod_count, false, pool_));
        Pointer temp(allocate_uint(coeff_mod_count, pool_));
        divide_uint_uint(context.total_coeff_modulus().data(), wide_plain_modulus.get(), coeff_mod_count, coeff_div_plain_modulus_.get(), temp.get(), pool_);

        // Calculate (plain_modulus + 1) / 2.
        plain_upper_half_threshold_ = (parms_.plain_modulus().value() + 1) >> 1;

        // Calculate coeff_modulus - plain_modulus.
        plain_upper_half_increment_ = allocate_uint(coeff_mod_count, pool_);
        sub_uint_uint(context.total_coeff_modulus().data(), wide_plain_modulus.get(), coeff_mod_count, plain_upper_half_increment_.get());

        // Calculate coeff_modulus[i] - plain_modulus if enable_fast_plain_lift
        if (qualifiers_.enable_fast_plain_lift)
        {
            plain_upper_half_increment_array_.resize(coeff_mod_count);
            for (int i = 0; i < coeff_mod_count; i++)
            {
                plain_upper_half_increment_array_[i] = coeff_modulus_[i].value() - parms_.plain_modulus().value();
            }
        }

        // Calculate upper_half_increment.
        upper_half_increment_ = allocate_uint(coeff_mod_count, pool_);
        multiply_truncate_uint_uint(wide_plain_modulus.get(), coeff_div_plain_modulus_.get(), coeff_mod_count, upper_half_increment_.get());
        sub_uint_uint(context.total_coeff_modulus().data(), upper_half_increment_.get(), coeff_mod_count, upper_half_increment_.get());

        // Decompose coeff_div_plain_modulus and upper_half_increment
        Pointer temp_reduction(allocate_uint(coeff_mod_count, pool_));
        for (int i = 0; i < coeff_mod_count; i++)
        {
            temp_reduction[i] = modulo_uint(coeff_div_plain_modulus_.get(), coeff_mod_count, coeff_modulus_[i], pool_);
        }
        set_uint_uint(temp_reduction.get(), coeff_mod_count, coeff_div_plain_modulus_.get());
        for (int i = 0; i < coeff_mod_count; i++)
        {
            temp_reduction[i] = modulo_uint(upper_half_increment_.get(), coeff_mod_count, coeff_modulus_[i], pool_);
        }
        set_uint_uint(temp_reduction.get(), coeff_mod_count, upper_half_increment_.get());

        // Calculate coeff_modulus_ / 2.
        coeff_modulus_div_two_ = allocate_uint(coeff_mod_count, pool_);
        right_shift_uint(context.total_coeff_modulus().data(), 1, coeff_mod_count, coeff_modulus_div_two_.get());

        // Set the big coeff modulus for noise computation
        product_modulus_ = allocate_uint(coeff_mod_count, pool_);
        set_uint_uint(context.total_coeff_modulus().data(), coeff_mod_count, product_modulus_.get());

        // Initialize moduli.
        mod_ = Modulus(product_modulus_.get(), coeff_mod_count);
        polymod_ = PolyModulus(parms_.poly_modulus().data(), coeff_count, poly_coeff_uint64_count);

        // Calculate map from Zmstar to generator representation
        populate_Zmstar_to_generator();
    }

    Evaluator::Evaluator(const Evaluator &copy) :
        pool_(copy.pool_), parms_(copy.parms_), qualifiers_(copy.qualifiers_),
        base_converter_(copy.base_converter_),
        coeff_small_ntt_tables_(copy.coeff_small_ntt_tables_),
        bsk_small_ntt_tables_(copy.bsk_small_ntt_tables_),
        plain_upper_half_threshold_(copy.plain_upper_half_threshold_),
        plain_upper_half_increment_array_(copy.plain_upper_half_increment_array_),
        coeff_modulus_(copy.coeff_modulus_),
        bsk_mod_array_(copy.bsk_mod_array_),
        inv_coeff_products_mod_coeff_array_(copy.inv_coeff_products_mod_coeff_array_),
        bsk_base_mod_count_(copy.bsk_base_mod_count_),
        Zmstar_to_generator_(copy.Zmstar_to_generator_)
    {
        int coeff_count = parms_.poly_modulus().coeff_count();
        int poly_coeff_uint64_count = parms_.poly_modulus().coeff_uint64_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());

        // Allocate memory and copy over values
        // Calculate upper_half_increment.
        upper_half_increment_ = allocate_uint(coeff_mod_count, pool_);
        set_uint_uint(copy.upper_half_increment_.get(), coeff_mod_count, upper_half_increment_.get());

        // Calculate coeff_modulus / plain_modulus.
        coeff_div_plain_modulus_ = allocate_uint(coeff_mod_count, pool_);
        set_uint_uint(copy.coeff_div_plain_modulus_.get(), coeff_mod_count, coeff_div_plain_modulus_.get());

        // Calculate coeff_modulus - plain_modulus.
        plain_upper_half_increment_ = allocate_uint(coeff_mod_count, pool_);
        set_uint_uint(copy.plain_upper_half_increment_.get(), coeff_mod_count, plain_upper_half_increment_.get());

        // Calculate coeff_modulus_ / 2.
        coeff_modulus_div_two_ = allocate_uint(coeff_mod_count, pool_);
        set_uint_uint(copy.coeff_div_plain_modulus_.get(), coeff_mod_count, coeff_div_plain_modulus_.get());

        // Populate coeff products array for compose functions (used in noise budget)
        coeff_products_array_ = allocate_uint(coeff_mod_count * coeff_mod_count, pool_);
        set_uint_uint(copy.coeff_products_array_.get(), coeff_mod_count * coeff_mod_count, coeff_products_array_.get());

        // Set the big coeff modulus for noise computation
        product_modulus_ = allocate_uint(coeff_mod_count, pool_);
        set_uint_uint(copy.product_modulus_.get(), coeff_mod_count, product_modulus_.get());

        // Initialize moduli.
        mod_ = Modulus(product_modulus_.get(), coeff_mod_count);
        polymod_ = PolyModulus(parms_.poly_modulus().data(), coeff_count, poly_coeff_uint64_count);
    }

    void Evaluator::compose(uint64_t *value, const MemoryPoolHandle &pool)
    {
#ifdef SEAL_DEBUG
        if (value == nullptr)
        {
            throw invalid_argument("value cannot be null");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
#endif
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int total_uint64_count = coeff_mod_count * coeff_count;

        Pointer coefficients(allocate_uint(total_uint64_count, pool));
        uint64_t *coefficients_ptr = coefficients.get();

        // Re-merge the coefficients first
        for (int i = 0; i < coeff_count; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                coefficients_ptr[(i * coeff_mod_count) + j] = value[(j * coeff_count) + i];
            }
        }

        Pointer temp(allocate_uint(coeff_mod_count, pool));
        set_zero_uint(total_uint64_count, value);

        for (int i = 0; i < coeff_count; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                uint64_t tmp = multiply_uint_uint_mod(coefficients_ptr[j], inv_coeff_products_mod_coeff_array_[j], coeff_modulus_[j]);
                multiply_uint_uint64(coeff_products_array_.get() + (j * coeff_mod_count), coeff_mod_count, tmp, coeff_mod_count, temp.get());
                add_uint_uint_mod(temp.get(), value + (i * coeff_mod_count), mod_.get(), coeff_mod_count, value + (i * coeff_mod_count));
            }
            set_zero_uint(coeff_mod_count, temp.get());
            coefficients_ptr += coeff_mod_count;
        }
    }

    void Evaluator::populate_Zmstar_to_generator()
    {
        uint64_t n = parms_.poly_modulus().coeff_count() - 1;
        uint64_t m = n << 1;

        for (uint64_t i = 0; i < n / 2; i++)
        {
            uint64_t galois_elt = (exponentiate_uint64(3, i)) & (m - 1);
            pair<uint64_t, uint64_t> temp_pair1{ i, 0 };
            Zmstar_to_generator_.emplace(galois_elt, temp_pair1);
            galois_elt = (exponentiate_uint64(3, i) * (m - 1)) & (m - 1);
            pair<uint64_t, uint64_t> temp_pair2 = { i, 1 };
            Zmstar_to_generator_.emplace(galois_elt, temp_pair2);
        }
    }

    void Evaluator::negate(Ciphertext &encrypted)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted_size = encrypted.size();

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Negate each poly in the array
        for (int j = 0; j < encrypted_size; j++)
        {
            for (int i = 0; i < coeff_mod_count; i++)
            {
                negate_poly_coeffmod(encrypted.data(j) + (i * coeff_count), 
                    coeff_count, coeff_modulus_[i], encrypted.data(j) + (i * coeff_count));
            }
        }
    }

    void Evaluator::add(Ciphertext &encrypted1, const Ciphertext &encrypted2)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted1_size = encrypted1.size();
        int encrypted2_size = encrypted2.size();
        int max_count = max(encrypted1_size, encrypted2_size);
        int min_count = min(encrypted1_size, encrypted2_size);

        // Verify parameters.
        if (encrypted1.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (encrypted2.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }

        // Prepare destination
        encrypted1.resize(parms_, max_count);

        // Add ciphertexts
        for (int j = 0; j < min_count; j++)
        {
            for (int i = 0; i < coeff_mod_count; i++)
            {
                add_poly_poly_coeffmod(encrypted1.data(j) + (i * coeff_count), 
                    encrypted2.data(j) + (i * coeff_count), coeff_count, coeff_modulus_[i], 
                    encrypted1.data(j) + (i * coeff_count));
            }
        }

        // Copy the remainding polys of the array with larger count into encrypted1
        if (encrypted1_size < encrypted2_size)
        {
            set_poly_poly(encrypted2.data(min_count), coeff_count * (encrypted2_size - encrypted1_size),
                coeff_mod_count, encrypted1.data(encrypted1_size));
        }
    }

    void Evaluator::add_many(const vector<Ciphertext> &encrypteds, Ciphertext &destination)
    {
        if (encrypteds.empty())
        {
            throw invalid_argument("encrypteds cannot be empty");
        }

        destination = encrypteds[0];
        for (size_t i = 1; i < encrypteds.size(); i++)
        {
            add(destination, encrypteds[i]);
        }
    }

    void Evaluator::sub(Ciphertext &encrypted1, const Ciphertext &encrypted2)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted1_size = encrypted1.size();
        int encrypted2_size = encrypted2.size();
        int max_count = max(encrypted1_size, encrypted2_size);
        int min_count = min(encrypted1_size, encrypted2_size);

        // Verify parameters.
        if (encrypted1.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (encrypted2.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }

        // Prepare destination
        encrypted1.resize(parms_, max_count);

        // Subtract polynomials.
        for (int j = 0; j < min_count; j++)
        {
            for (int i = 0; i < coeff_mod_count; i++)
            {
                sub_poly_poly_coeffmod(encrypted1.data(j) + (i * coeff_count),
                    encrypted2.data(j) + (i * coeff_count), coeff_count, coeff_modulus_[i], 
                    encrypted1.data(j) + (i * coeff_count));
            }
        }

        // If encrypted2 has larger count, negate remaining entries
        if (encrypted1_size < encrypted2_size)
        {
            for (int i = 0; i < coeff_mod_count; i++)
            {
                negate_poly_coeffmod(encrypted2.data(encrypted1_size) + (i * coeff_count),
                    coeff_count * (encrypted2_size - encrypted1_size), coeff_modulus_[i],
                    encrypted1.data(encrypted1_size) + (i * coeff_count));
            }
        }
    }

    void Evaluator::multiply(Ciphertext &encrypted1, const Ciphertext &encrypted2, const MemoryPoolHandle &pool)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int bsk_mtilde_count = bsk_base_mod_count_ + 1;
        int encrypted1_size = encrypted1.size();
        int encrypted2_size = encrypted2.size();

        // Verify parameters.
        if (encrypted1.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted1 is not valid for encryption parameters");
        }
        if (encrypted2.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted2 is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Determine destination.size()
        // Default is 3 (c_0, c_1, c_2)
        int dest_count = encrypted1_size + encrypted2_size - 1;

        // Prepare destination
        encrypted1.resize(parms_, dest_count);

        int encrypted_ptr_increment = coeff_count * coeff_mod_count;
        int encrypted_bsk_mtilde_ptr_increment = coeff_count * bsk_mtilde_count;
        int encrypted_bsk_ptr_increment = coeff_count * bsk_base_mod_count_;

        // Make temp polys for FastBConverter result from q ---> Bsk U {m_tilde}
        Pointer tmp_encrypted1_bsk_mtilde(allocate_poly(coeff_count * encrypted1_size, bsk_mtilde_count, pool));
        Pointer tmp_encrypted2_bsk_mtilde(allocate_poly(coeff_count * encrypted2_size, bsk_mtilde_count, pool));

        // Make temp polys for FastBConverter result from Bsk U {m_tilde} -----> Bsk
        Pointer tmp_encrypted1_bsk(allocate_poly(coeff_count * encrypted1_size, bsk_base_mod_count_, pool));
        Pointer tmp_encrypted2_bsk(allocate_poly(coeff_count * encrypted2_size, bsk_base_mod_count_, pool));

        // Step 0: fast base convert from q to Bsk U {m_tilde}
        // Step 1: reduce q-overflows in Bsk
        // Iterate over all the ciphertexts inside encrypted1
        for (int i = 0; i < encrypted1_size; i++)
        {
            base_converter_.fastbconv_mtilde(encrypted1.data(i), 
                tmp_encrypted1_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment), pool);
            base_converter_.mont_rq(tmp_encrypted1_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment), 
                tmp_encrypted1_bsk.get() + (i * encrypted_bsk_ptr_increment));
        }
        
        // Iterate over all the ciphertexts inside encrypted2
        for (int i = 0; i < encrypted2_size; i++)
        {
            base_converter_.fastbconv_mtilde(encrypted2.data(i), 
                tmp_encrypted2_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment), pool);
            base_converter_.mont_rq(tmp_encrypted2_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment), 
                tmp_encrypted2_bsk.get() + (i * encrypted_bsk_ptr_increment));
        }
        
        // Step 2: compute product and multiply plain modulus to the result
        // We need to multiply both in q and Bsk. Values in encrypted_safe are in base q and values in tmp_encrypted_bsk are in base Bsk
        // We iterate over destination poly array and generate each poly based on the indices of inputs (arbitrary sizes for ciphertexts)
        // First allocate two temp polys: one for results in base q and the other for the result in base Bsk
        // These need to be zero for the arbitrary size multiplication; not for 2x2 though
        Pointer tmp_des_coeff_base(allocate_zero_poly(coeff_count * dest_count, coeff_mod_count, pool));
        Pointer tmp_des_bsk_base(allocate_zero_poly(coeff_count * dest_count, bsk_base_mod_count_, pool));

        // Allocate two tmp polys: one for NTT multiplication results in base q and one for result in base Bsk
        Pointer tmp1_poly_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));
        Pointer tmp1_poly_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count_, pool));
        Pointer tmp2_poly_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));
        Pointer tmp2_poly_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count_, pool));

        int current_encrypted1_limit = 0;

        // First convert all the inputs into NTT form
        Pointer copy_encrypted1_ntt_coeff_mod(allocate_poly(coeff_count * encrypted1_size, coeff_mod_count, pool));
        set_poly_poly(encrypted1.data(), coeff_count * encrypted1_size, coeff_mod_count, copy_encrypted1_ntt_coeff_mod.get());

        Pointer copy_encrypted1_ntt_bsk_base_mod(allocate_poly(coeff_count * encrypted1_size, bsk_base_mod_count_, pool));
        set_poly_poly(tmp_encrypted1_bsk.get(), coeff_count * encrypted1_size, bsk_base_mod_count_, copy_encrypted1_ntt_bsk_base_mod.get());

        Pointer copy_encrypted2_ntt_coeff_mod(allocate_poly(coeff_count * encrypted2_size, coeff_mod_count, pool));
        set_poly_poly(encrypted2.data(), coeff_count * encrypted2_size, coeff_mod_count, copy_encrypted2_ntt_coeff_mod.get());

        Pointer copy_encrypted2_ntt_bsk_base_mod(allocate_poly(coeff_count * encrypted2_size, bsk_base_mod_count_, pool));
        set_poly_poly(tmp_encrypted2_bsk.get(), coeff_count * encrypted2_size, bsk_base_mod_count_, copy_encrypted2_ntt_bsk_base_mod.get());

        for (int i = 0; i < encrypted1_size; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted1_ntt_coeff_mod.get() + (j * coeff_count) + (i * encrypted_ptr_increment), coeff_small_ntt_tables_[j]);
            }
            for (int j = 0; j < bsk_base_mod_count_; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted1_ntt_bsk_base_mod.get() + (j * coeff_count) + (i * encrypted_bsk_ptr_increment), bsk_small_ntt_tables_[j]);
            }
        }

        for (int i = 0; i < encrypted2_size; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted2_ntt_coeff_mod.get() + (j * coeff_count) + (i * encrypted_ptr_increment), coeff_small_ntt_tables_[j]);
            }
            for (int j = 0; j < bsk_base_mod_count_; j++)
            {
                // Lazy reduction
                ntt_negacyclic_harvey_lazy(copy_encrypted2_ntt_bsk_base_mod.get() + (j * coeff_count) + (i * encrypted_bsk_ptr_increment), bsk_small_ntt_tables_[j]);
            }
        }

        // Perform Karatsuba multiplication on size 2 ciphertexts
        if (encrypted1_size == 2 && encrypted2_size == 2)
        {
            Pointer tmp_first_mul_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));

            // Compute c0 + c1 and c0*d0 in base q
            uint64_t *temp_ptr_1 = tmp1_poly_coeff_base.get();
            uint64_t *temp_ptr_2 = copy_encrypted1_ntt_coeff_mod.get();
            uint64_t *temp_ptr_3 = temp_ptr_2 + encrypted_ptr_increment;
            for (int i = 0; i < coeff_mod_count; i++)
            {
                //add_poly_poly_coeffmod(copy_encrypted1_ntt_coeff_mod.get() + (i * coeff_count), 
                //    copy_encrypted1_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment, 
                //    coeff_count, coeff_modulus_[i], tmp1_poly_coeff_base.get() + (i * coeff_count));

                // Lazy reduction
                for (int j = 0; j < coeff_count; j++)
                {
                    *temp_ptr_1++ = *temp_ptr_2++ + *temp_ptr_3++;
                }
                dyadic_product_coeffmod(copy_encrypted1_ntt_coeff_mod.get() + (i * coeff_count), 
                    copy_encrypted2_ntt_coeff_mod.get() + (i * coeff_count), coeff_count, coeff_modulus_[i], 
                    tmp_first_mul_coeff_base.get() + (i * coeff_count));
            }

            Pointer tmp_first_mul_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count_, pool));

            // Compute c0 + c1 and c0*d0 in base bsk
            temp_ptr_1 = tmp1_poly_bsk_base.get();
            temp_ptr_2 = copy_encrypted1_ntt_bsk_base_mod.get();
            temp_ptr_3 = temp_ptr_2 + encrypted_bsk_ptr_increment;
            for (int i = 0; i < bsk_base_mod_count_; i++)
            {
                //add_poly_poly_coeffmod(copy_encrypted1_ntt_bsk_base_mod.get() + (i * coeff_count), 
                //    copy_encrypted1_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment, 
                //    coeff_count, bsk_mod_array_[i], tmp1_poly_bsk_base.get() + (i * coeff_count));
                for (int j = 0; j < coeff_count; j++)
                {
                    *temp_ptr_1++ = *temp_ptr_2++ + *temp_ptr_3++;
                }
                dyadic_product_coeffmod(copy_encrypted1_ntt_bsk_base_mod.get() + (i * coeff_count), 
                    copy_encrypted2_ntt_bsk_base_mod.get() + (i * coeff_count), coeff_count, bsk_mod_array_[i], 
                    tmp_first_mul_bsk_base.get() + (i * coeff_count));
            }

            Pointer tmp_second_mul_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));

            // Compute d0 + d1 and c1*d1 in base q
            temp_ptr_1 = tmp2_poly_coeff_base.get();
            temp_ptr_2 = copy_encrypted2_ntt_coeff_mod.get();
            temp_ptr_3 = temp_ptr_2 + encrypted_ptr_increment;
            for (int i = 0; i < coeff_mod_count; i++)
            {
                //add_poly_poly_coeffmod(copy_encrypted2_ntt_coeff_mod.get() + (i * coeff_count), 
                //    copy_encrypted2_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment, 
                //    coeff_count, coeff_modulus_[i], tmp2_poly_coeff_base.get() + (i * coeff_count));
                for (int j = 0; j < coeff_count; j++)
                {
                    *temp_ptr_1++ = *temp_ptr_2++ + *temp_ptr_3++;
                }
                dyadic_product_coeffmod(copy_encrypted1_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment, 
                    copy_encrypted2_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment, 
                    coeff_count, coeff_modulus_[i], tmp_second_mul_coeff_base.get() + (i * coeff_count));
            }

            Pointer tmp_second_mul_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count_, pool));

            // Compute d0 + d1 and c1*d1 in base bsk
            temp_ptr_1 = tmp2_poly_bsk_base.get();
            temp_ptr_2 = copy_encrypted2_ntt_bsk_base_mod.get();
            temp_ptr_3 = temp_ptr_2 + encrypted_bsk_ptr_increment;
            for (int i = 0; i < bsk_base_mod_count_; i++)
            {
                //add_poly_poly_coeffmod(copy_encrypted2_ntt_bsk_base_mod.get() + (i * coeff_count), 
                //    copy_encrypted2_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment, 
                //    coeff_count, bsk_mod_array_[i], tmp2_poly_bsk_base.get() + (i * coeff_count));
                for (int j = 0; j < coeff_count; j++)
                {
                    *temp_ptr_1++ = *temp_ptr_2++ + *temp_ptr_3++;
                }
                dyadic_product_coeffmod(copy_encrypted1_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment, 
                    copy_encrypted2_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment, 
                    coeff_count, bsk_mod_array_[i], tmp_second_mul_bsk_base.get() + (i * coeff_count));
            }

            Pointer tmp_mul_poly_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));
            Pointer tmp_mul_poly_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count_, pool));

            // Set destination first and third polys in base q
            // Des[0] in base q
            set_poly_poly(tmp_first_mul_coeff_base.get(), coeff_count, coeff_mod_count, tmp_des_coeff_base.get());

            // Des[2] in base q
            set_poly_poly(tmp_second_mul_coeff_base.get(), coeff_count, coeff_mod_count, tmp_des_coeff_base.get() + 2 * encrypted_ptr_increment);
            
            // Compute (c0 + c1)*(d0 + d1) - c0*d0 - c1*d1 in base q
            for (int i = 0; i < coeff_mod_count; i++)
            {
                dyadic_product_coeffmod(tmp1_poly_coeff_base.get() + (i * coeff_count), tmp2_poly_coeff_base.get() + (i * coeff_count), 
                    coeff_count, coeff_modulus_[i], tmp_mul_poly_coeff_base.get() + (i * coeff_count));
                sub_poly_poly_coeffmod(tmp_mul_poly_coeff_base.get() + (i * coeff_count), 
                    tmp_first_mul_coeff_base.get() + (i * coeff_count), coeff_count, coeff_modulus_[i], 
                    tmp_mul_poly_coeff_base.get() + (i * coeff_count));
                
                // Des[1] in base q
                sub_poly_poly_coeffmod(tmp_mul_poly_coeff_base.get() + (i * coeff_count), 
                    tmp_second_mul_coeff_base.get() + (i * coeff_count), coeff_count, coeff_modulus_[i], 
                    tmp_des_coeff_base.get() + (i * coeff_count) + encrypted_ptr_increment);
            }

            // Set destination first and third polys in base bsk
            // Des[0] in base bsk
            set_poly_poly(tmp_first_mul_bsk_base.get(), coeff_count, bsk_base_mod_count_, tmp_des_bsk_base.get());

            // Des[2] in base q
            set_poly_poly(tmp_second_mul_bsk_base.get(), coeff_count, bsk_base_mod_count_, 
                tmp_des_bsk_base.get() + 2 * encrypted_bsk_ptr_increment);

            // Compute (c0 + c1)*(d0 + d1)  - c0d0 - c1d1 in base bsk
            for (int i = 0; i < bsk_base_mod_count_; i++)
            {
                dyadic_product_coeffmod(tmp1_poly_bsk_base.get() + (i * coeff_count), 
                    tmp2_poly_bsk_base.get() + (i * coeff_count), coeff_count, bsk_mod_array_[i], 
                    tmp_mul_poly_bsk_base.get() + (i * coeff_count));
                sub_poly_poly_coeffmod(tmp_mul_poly_bsk_base.get() + (i * coeff_count), 
                    tmp_first_mul_bsk_base.get() + (i * coeff_count), coeff_count, bsk_mod_array_[i], 
                    tmp_mul_poly_bsk_base.get() + (i * coeff_count));

                // Des[1] in bsk
                sub_poly_poly_coeffmod(tmp_mul_poly_bsk_base.get() + (i * coeff_count), 
                    tmp_second_mul_bsk_base.get() + (i * coeff_count), coeff_count, bsk_mod_array_[i], 
                    tmp_des_bsk_base.get() + (i * coeff_count) + encrypted_bsk_ptr_increment); 
            }
        }
        else
        {
            // Perform multiplication on arbitrary size ciphertexts
            for (int secret_power_index = 0; secret_power_index < dest_count; secret_power_index++)
            {
                // Loop over encrypted1 components [i], seeing if a match exists with an encrypted2 
                // component [j] such that [i+j]=[secret_power_index]
                // Only need to check encrypted1 components up to and including [secret_power_index], 
                // and strictly less than [encrypted_array.size()]
                current_encrypted1_limit = min(encrypted1_size, secret_power_index + 1);

                for (int encrypted1_index = 0; encrypted1_index < current_encrypted1_limit; encrypted1_index++)
                {
                    // check if a corresponding component in encrypted2 exists
                    if (encrypted2_size > secret_power_index - encrypted1_index)
                    {
                        int encrypted2_index = secret_power_index - encrypted1_index;

                        // NTT Multiplication and addition for results in q
                        for (int i = 0; i < coeff_mod_count; i++)
                        {
                            dyadic_product_coeffmod(copy_encrypted1_ntt_coeff_mod.get() + (i * coeff_count) + (encrypted_ptr_increment * encrypted1_index), 
                                copy_encrypted2_ntt_coeff_mod.get() + (i * coeff_count) + (encrypted_ptr_increment * encrypted2_index), 
                                coeff_count, coeff_modulus_[i], tmp1_poly_coeff_base.get() + (i * coeff_count));
                            add_poly_poly_coeffmod(tmp1_poly_coeff_base.get() + (i * coeff_count), 
                                tmp_des_coeff_base.get() + (i * coeff_count) + (secret_power_index * coeff_count * coeff_mod_count), coeff_count, 
                                coeff_modulus_[i], tmp_des_coeff_base.get() + (i * coeff_count) + (secret_power_index * coeff_count * coeff_mod_count));
                        }

                        // NTT Multiplication and addition for results in Bsk
                        for (int i = 0; i < bsk_base_mod_count_; i++)
                        {
                            dyadic_product_coeffmod(copy_encrypted1_ntt_bsk_base_mod.get() + (i * coeff_count) + (encrypted_bsk_ptr_increment * encrypted1_index), 
                                copy_encrypted2_ntt_bsk_base_mod.get() + (i * coeff_count) + (encrypted_bsk_ptr_increment * encrypted2_index), 
                                coeff_count, bsk_mod_array_[i], tmp1_poly_bsk_base.get() + (i * coeff_count));
                            add_poly_poly_coeffmod(tmp1_poly_bsk_base.get() + (i * coeff_count), 
                                tmp_des_bsk_base.get() + (i * coeff_count) + (secret_power_index * coeff_count * bsk_base_mod_count_), 
                                coeff_count, bsk_mod_array_[i], 
                                tmp_des_bsk_base.get() + (i * coeff_count) + (secret_power_index * coeff_count * bsk_base_mod_count_));
                        }
                    }
                }
            }
        }
        // Convert back outputs from NTT form
        for (int i = 0; i < dest_count; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey(tmp_des_coeff_base.get() + (i * (encrypted_ptr_increment)) + (j * coeff_count), coeff_small_ntt_tables_[j]);
            }
            for (int j = 0; j < bsk_base_mod_count_; j++)
            {
                inverse_ntt_negacyclic_harvey(tmp_des_bsk_base.get() + (i * (encrypted_bsk_ptr_increment)) + (j * coeff_count), bsk_small_ntt_tables_[j]);
            }
        }

        // Now we multiply plain modulus to both results in base q and Bsk and allocate them together in one 
        // container as (te0)q(te'0)Bsk | ... |te count)q (te' count)Bsk to make it ready for fast_floor 
        Pointer tmp_coeff_bsk_together(allocate_poly(coeff_count, dest_count * (coeff_mod_count + bsk_base_mod_count_), pool));
        uint64_t *tmp_coeff_bsk_together_ptr = tmp_coeff_bsk_together.get();

        // Base q 
        for (int i = 0; i < dest_count; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                multiply_poly_scalar_coeffmod(tmp_des_coeff_base.get() + (j * coeff_count) + (i * encrypted_ptr_increment), 
                    coeff_count, parms_.plain_modulus().value(), coeff_modulus_[j], tmp_coeff_bsk_together_ptr + (j * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_ptr_increment;
            
            for (int k = 0; k < bsk_base_mod_count_; k++)
            {
                multiply_poly_scalar_coeffmod(tmp_des_bsk_base.get() + (k * coeff_count) + (i * encrypted_bsk_ptr_increment), 
                    coeff_count, parms_.plain_modulus().value(), bsk_mod_array_[k], tmp_coeff_bsk_together_ptr + (k * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_bsk_ptr_increment;
        }

        // Allocate a new poly for fast floor result in Bsk
        Pointer tmp_result_bsk(allocate_poly(coeff_count, dest_count * bsk_base_mod_count_, pool));
        for (int i = 0; i < dest_count; i++)
        {
            // Step 3: fast floor from q U {Bsk} to Bsk 
            base_converter_.fast_floor(tmp_coeff_bsk_together.get() + (i * (encrypted_ptr_increment + encrypted_bsk_ptr_increment)), 
                tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment), pool);

            // Step 4: fast base convert from Bsk to q
            base_converter_.fastbconv_sk(tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment), encrypted1.data(i), pool);
        }
    }

    void Evaluator::square(Ciphertext &encrypted, const MemoryPoolHandle &pool)
    {
        int encrypted_size = encrypted.size();

        // Optimization implemented currently only for size 2 ciphertexts
        if (encrypted_size != 2)
        {
            multiply(encrypted, encrypted, pool);
            return;
        }

        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int bsk_mtilde_count = bsk_base_mod_count_ + 1;
        int encrypted_ptr_increment = coeff_count * coeff_mod_count;
        int encrypted_bsk_mtilde_ptr_increment = coeff_count * bsk_mtilde_count;
        int encrypted_bsk_ptr_increment = coeff_count * bsk_base_mod_count_;

        // Determine destination_array.size()
        int dest_count = (encrypted_size << 1) - 1;

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Prepare destination
        encrypted.resize(parms_, dest_count);

        // Make temp poly for FastBConverter result from q ---> Bsk U {m_tilde}
        Pointer tmp_encrypted_bsk_mtilde(allocate_poly(coeff_count * encrypted_size, bsk_mtilde_count, pool));

        // Make temp poly for FastBConverter result from Bsk U {m_tilde} -----> Bsk
        Pointer tmp_encrypted_bsk(allocate_poly(coeff_count * encrypted_size, bsk_base_mod_count_, pool));

        // Step 0: fast base convert from q to Bsk U {m_tilde}
        // Step 1: reduce q-overflows in Bsk
        // Iterate over all the ciphertexts inside encrypted1
        for (int i = 0; i < encrypted_size; i++)
        {
            base_converter_.fastbconv_mtilde(encrypted.data(i),
                tmp_encrypted_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment), pool);
            base_converter_.mont_rq(tmp_encrypted_bsk_mtilde.get() + (i * encrypted_bsk_mtilde_ptr_increment),
                tmp_encrypted_bsk.get() + (i * encrypted_bsk_ptr_increment));
        }

        // Step 2: compute product and multiply plain modulus to the result
        // We need to multiply both in q and Bsk. Values in encrypted_safe are in base q and values in 
        // tmp_encrypted_bsk are in base Bsk
        // We iterate over destination poly array and generate each poly based on the indices of inputs 
        // (arbitrary sizes for ciphertexts)
        // First allocate two temp polys: one for results in base q and the other for the result in base Bsk
        Pointer tmp_des_coeff_base(allocate_poly(coeff_count * dest_count, coeff_mod_count, pool));
        Pointer tmp_des_bsk_base(allocate_poly(coeff_count * dest_count, bsk_base_mod_count_, pool));

        // First convert all the inputs into NTT form
        Pointer copy_encrypted_ntt_coeff_mod(allocate_poly(coeff_count * encrypted_size, coeff_mod_count, pool));
        set_poly_poly(encrypted.data(), coeff_count * encrypted_size, coeff_mod_count, copy_encrypted_ntt_coeff_mod.get());

        Pointer copy_encrypted_ntt_bsk_base_mod(allocate_poly(coeff_count * encrypted_size, bsk_base_mod_count_, pool));
        set_poly_poly(tmp_encrypted_bsk.get(), coeff_count * encrypted_size, bsk_base_mod_count_, copy_encrypted_ntt_bsk_base_mod.get());

        for (int i = 0; i < encrypted_size; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                ntt_negacyclic_harvey_lazy(copy_encrypted_ntt_coeff_mod.get() + (j * coeff_count) + (i * encrypted_ptr_increment), coeff_small_ntt_tables_[j]);
            }
            for (int j = 0; j < bsk_base_mod_count_; j++)
            {
                ntt_negacyclic_harvey_lazy(copy_encrypted_ntt_bsk_base_mod.get() + (j * coeff_count) + (i * encrypted_bsk_ptr_increment), bsk_small_ntt_tables_[j]);
            }
        }

        // Perform fast squaring
        // Compute c0^2 in base q
        for (int i = 0; i < coeff_mod_count; i++)
        {
            // Des[0] in q
            dyadic_product_coeffmod(copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count), coeff_count, coeff_modulus_[i],
                tmp_des_coeff_base.get() + (i * coeff_count));

            // Des[2] in q
            dyadic_product_coeffmod(copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment,
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment, coeff_count,
                coeff_modulus_[i], tmp_des_coeff_base.get() + (i * coeff_count) + (2 * encrypted_ptr_increment));
        }

        // Compute c0^2 in base bsk
        for (int i = 0; i < bsk_base_mod_count_; i++)
        {
            // Des[0] in bsk
            dyadic_product_coeffmod(copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count), coeff_count, bsk_mod_array_[i],
                tmp_des_bsk_base.get() + (i * coeff_count));

            // Des[2] in bsk
            dyadic_product_coeffmod(copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment,
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment, coeff_count,
                bsk_mod_array_[i], tmp_des_bsk_base.get() + (i * coeff_count) + (2 * encrypted_bsk_ptr_increment));
        }

        Pointer tmp_second_mul_coeff_base(allocate_poly(coeff_count, coeff_mod_count, pool));

        // Compute 2*c0*c1 in base q
        for (int i = 0; i < coeff_mod_count; i++)
        {
            dyadic_product_coeffmod(copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_coeff_mod.get() + (i * coeff_count) + encrypted_ptr_increment, coeff_count,
                coeff_modulus_[i], tmp_second_mul_coeff_base.get() + (i * coeff_count));
            add_poly_poly_coeffmod(tmp_second_mul_coeff_base.get() + (i * coeff_count),
                tmp_second_mul_coeff_base.get() + (i * coeff_count), coeff_count, coeff_modulus_[i],
                tmp_des_coeff_base.get() + (i * coeff_count) + encrypted_ptr_increment);
        }

        Pointer tmp_second_mul_bsk_base(allocate_poly(coeff_count, bsk_base_mod_count_, pool));

        // Compute 2*c0*c1 in base bsk
        for (int i = 0; i < bsk_base_mod_count_; i++)
        {
            dyadic_product_coeffmod(copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count),
                copy_encrypted_ntt_bsk_base_mod.get() + (i * coeff_count) + encrypted_bsk_ptr_increment,
                coeff_count, bsk_mod_array_[i], tmp_second_mul_bsk_base.get() + (i * coeff_count));
            add_poly_poly_coeffmod(tmp_second_mul_bsk_base.get() + (i * coeff_count),
                tmp_second_mul_bsk_base.get() + (i * coeff_count), coeff_count, bsk_mod_array_[i],
                tmp_des_bsk_base.get() + (i * coeff_count) + encrypted_bsk_ptr_increment);
        }

        // Convert back outputs from NTT form
        for (int i = 0; i < dest_count; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey_lazy(tmp_des_coeff_base.get() + (i * (encrypted_ptr_increment)) + (j * coeff_count),
                    coeff_small_ntt_tables_[j]);
            }
            for (int j = 0; j < bsk_base_mod_count_; j++)
            {
                inverse_ntt_negacyclic_harvey_lazy(tmp_des_bsk_base.get() + (i * (encrypted_bsk_ptr_increment)) + (j * coeff_count), bsk_small_ntt_tables_[j]);
            }
        }

        // Now we multiply plain modulus to both results in base q and Bsk and allocate them together in one 
        // container as (te0)q(te'0)Bsk | ... |te count)q (te' count)Bsk to make it ready for fast_floor 
        Pointer tmp_coeff_bsk_together(allocate_poly(coeff_count, dest_count * (coeff_mod_count + bsk_base_mod_count_), pool));
        uint64_t *tmp_coeff_bsk_together_ptr = tmp_coeff_bsk_together.get();

        // Base q 
        for (int i = 0; i < dest_count; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                multiply_poly_scalar_coeffmod(tmp_des_coeff_base.get() + (j * coeff_count) + (i * encrypted_ptr_increment),
                    coeff_count, parms_.plain_modulus().value(), coeff_modulus_[j], tmp_coeff_bsk_together_ptr + (j * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_ptr_increment;

            for (int k = 0; k < bsk_base_mod_count_; k++)
            {
                multiply_poly_scalar_coeffmod(tmp_des_bsk_base.get() + (k * coeff_count) + (i * encrypted_bsk_ptr_increment),
                    coeff_count, parms_.plain_modulus().value(), bsk_mod_array_[k], tmp_coeff_bsk_together_ptr + (k * coeff_count));
            }
            tmp_coeff_bsk_together_ptr += encrypted_bsk_ptr_increment;
        }

        // Allocate a new poly for fast floor result in Bsk
        Pointer tmp_result_bsk(allocate_poly(coeff_count, dest_count * bsk_base_mod_count_, pool));
        for (int i = 0; i < dest_count; i++)
        {
            // Step 3: fast floor from q U {Bsk} to Bsk 
            base_converter_.fast_floor(tmp_coeff_bsk_together.get() + (i * (encrypted_ptr_increment + encrypted_bsk_ptr_increment)),
                tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment), pool);

            // Step 4: fast base convert from Bsk to q
            base_converter_.fastbconv_sk(tmp_result_bsk.get() + (i * encrypted_bsk_ptr_increment), encrypted.data(i), pool);
        }
    }

    void Evaluator::relinearize(Ciphertext &encrypted, const EvaluationKeys &evaluation_keys, int destination_size, const MemoryPoolHandle &pool)
    {
        // Extract encryption parameters.
        int encrypted_size = encrypted.size();

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (destination_size < 2 || destination_size > encrypted_size)
        {
            throw invalid_argument("destination_size must be greater than or equal to 2 and less than or equal to current count");
        }
        if (evaluation_keys.hash_block() != parms_.hash_block())
        {
            throw invalid_argument("evaluation_keys is not valid for encryption parameters");
        }
        if (evaluation_keys.size() < encrypted_size - 2)
        {
            throw invalid_argument("not enough evaluation keys");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // If encrypted is already at the desired level, return
        if (destination_size == encrypted_size)
        {
            return;
        }

        // Calculate number of relinearize_one_step calls needed
        int relins_needed = encrypted_size - destination_size;

        // Update temp to store the current result after relinearization
        for (int i = 0; i < relins_needed; i++)
        {
            relinearize_one_step(encrypted.data(), encrypted_size, evaluation_keys, pool);
            encrypted_size--;
        }

        // Put the output of final relinearization into destination.
        // Prepare destination only at this point because we are resizing down
        encrypted.resize(parms_, destination_size);
    }

    void Evaluator::relinearize_one_step(uint64_t *encrypted, int encrypted_size, const EvaluationKeys &evaluation_keys, const MemoryPoolHandle &pool)
    {
#ifdef SEAL_DEBUG
        if (encrypted == nullptr)
        {
            throw invalid_argument("encrypted cannot be null");
        }
        if (encrypted_size <= 2)
        {
            throw invalid_argument("encrypted_size must be at least 3");
        }
        if (evaluation_keys.hash_block() != parms_.hash_block())
        {
            throw invalid_argument("evaluation_keys is not valid for encryption parameters");
        }
        if (evaluation_keys.size() < encrypted_size - 2)
        {
            throw invalid_argument("not enough evaluation keys");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
#endif
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int array_poly_uint64_count = coeff_count * coeff_mod_count;

        const uint64_t *encrypted_coeff = encrypted + (encrypted_size - 1) * array_poly_uint64_count;
        Pointer encrypted_coeff_prod_inv_coeff(allocate_uint(coeff_count, pool));

        // Decompose encrypted_array[count-1] into base w
        // Want to create an array of polys, each of whose components i is (encrypted_array[count-1])^(i) - in the notation of FV paper
        // This allocation stores one of the decomposed factors modulo one of the primes
        Pointer decomp_encrypted_last(allocate_uint(coeff_count, pool));

        // Lazy reduction   
        Pointer wide_innerresult0(allocate_zero_poly(coeff_count, 2 * coeff_mod_count, pool));
        Pointer wide_innerresult1(allocate_zero_poly(coeff_count, 2 * coeff_mod_count, pool));
        Pointer innerresult(allocate_poly(coeff_count, coeff_mod_count, pool));
        Pointer temp_decomp_coeff(allocate_uint(coeff_count, pool));

        /*
        For lazy reduction to work here, we need to ensure that the 128-bit accumulators (wide_innerresult0 and wide_innerresult1)
        do not overflow. Since the modulus primes are at most 60 bits, if the total number of summands is K, then the size of the
        total sum of products (without reduction) is at most 62 + 60 + bit_length(K). We need this to be at most 128, thus we need
        bit_length(K) <= 6. Thus, we need K <= 63. In this case, this means sum_i evaluation_keys.data()[0][i].size() / 2 <= 63.
        */
        for (int i = 0; i < coeff_mod_count; i++)
        {
            multiply_poly_scalar_coeffmod(encrypted_coeff + (i * coeff_count), coeff_count, 
                inv_coeff_products_mod_coeff_array_[i], coeff_modulus_[i], encrypted_coeff_prod_inv_coeff.get());

            int shift = 0;
            const Ciphertext &key_component_ref = evaluation_keys.data()[0][i];
            int keys_size = key_component_ref.size();
            for (int k = 0; k < keys_size; k += 2)
            {
                const uint64_t *key_ptr_0 = key_component_ref.data(k);
                const uint64_t *key_ptr_1 = key_component_ref.data(k + 1);

                // Decompose here
                int decomposition_bit_count = evaluation_keys.decomposition_bit_count();
                for (int coeff_index = 0; coeff_index < coeff_count; coeff_index++)
                {
                    decomp_encrypted_last[coeff_index] = encrypted_coeff_prod_inv_coeff[coeff_index] >> shift;
                    decomp_encrypted_last[coeff_index] &= (1ULL << decomposition_bit_count) - 1;
                }

                uint64_t *wide_innerresult0_ptr = wide_innerresult0.get();
                uint64_t *wide_innerresult1_ptr = wide_innerresult1.get();
                for (int j = 0; j < coeff_mod_count; j++)
                {
                    uint64_t *temp_decomp_coeff_ptr = temp_decomp_coeff.get();
                    set_uint_uint(decomp_encrypted_last.get(), coeff_count, temp_decomp_coeff_ptr);

                    // We don't reduce here, so might get up to two extra bits. Thus 62 bits at most.
                    ntt_negacyclic_harvey_lazy(temp_decomp_coeff_ptr, coeff_small_ntt_tables_[j]);

                    // Lazy reduction
                    uint64_t wide_innerproduct[2];
                    for (int m = 0; m < coeff_count; m++, wide_innerresult0_ptr += 2)
                    {
                        multiply_uint64(*temp_decomp_coeff_ptr++, *key_ptr_0++, wide_innerproduct);
                        unsigned char carry = add_uint64(wide_innerresult0_ptr[0], wide_innerproduct[0], 0,
                            wide_innerresult0_ptr);
                        wide_innerresult0_ptr[1] += wide_innerproduct[1] + carry;
                    }

                    temp_decomp_coeff_ptr = temp_decomp_coeff.get();
                    for (int m = 0; m < coeff_count; m++, wide_innerresult1_ptr += 2)
                    {
                        multiply_uint64(*temp_decomp_coeff_ptr++, *key_ptr_1++, wide_innerproduct);
                        unsigned char carry = add_uint64(wide_innerresult1_ptr[0], wide_innerproduct[0], 0,
                            wide_innerresult1_ptr);
                        wide_innerresult1_ptr[1] += wide_innerproduct[1] + carry;
                    }
                }
                shift += decomposition_bit_count;
            }
        }

        uint64_t *innerresult_poly_ptr = innerresult.get();
        uint64_t *wide_innerresult_poly_ptr = wide_innerresult0.get();
        uint64_t *encrypted_ptr = encrypted;
        uint64_t *innerresult_coeff_ptr = innerresult_poly_ptr;
        uint64_t *wide_innerresult_coeff_ptr = wide_innerresult_poly_ptr;
        for (int i = 0; i < coeff_mod_count; i++, innerresult_poly_ptr += coeff_count,
            wide_innerresult_poly_ptr += 2 * coeff_count, encrypted_ptr += coeff_count)
        {
            for (int m = 0; m < coeff_count; m++, wide_innerresult_coeff_ptr += 2)
            {
                *innerresult_coeff_ptr++ = barrett_reduce_128(wide_innerresult_coeff_ptr, coeff_modulus_[i]);
            }
            inverse_ntt_negacyclic_harvey(innerresult_poly_ptr, coeff_small_ntt_tables_[i]);
            add_poly_poly_coeffmod(encrypted_ptr, innerresult_poly_ptr, coeff_count,
                coeff_modulus_[i], encrypted_ptr);
        }

        innerresult_poly_ptr = innerresult.get();
        wide_innerresult_poly_ptr = wide_innerresult1.get();
        encrypted_ptr = encrypted + array_poly_uint64_count;
        innerresult_coeff_ptr = innerresult_poly_ptr;
        wide_innerresult_coeff_ptr = wide_innerresult_poly_ptr;
        for (int i = 0; i < coeff_mod_count; i++, innerresult_poly_ptr += coeff_count,
            wide_innerresult_poly_ptr += 2 * coeff_count, encrypted_ptr += coeff_count)
        {
            for (int m = 0; m < coeff_count; m++, wide_innerresult_coeff_ptr += 2)
            {
                *innerresult_coeff_ptr++ = barrett_reduce_128(wide_innerresult_coeff_ptr, coeff_modulus_[i]);
            }
            inverse_ntt_negacyclic_harvey(innerresult_poly_ptr, coeff_small_ntt_tables_[i]);
            add_poly_poly_coeffmod(encrypted_ptr, innerresult_poly_ptr, coeff_count,
                coeff_modulus_[i], encrypted_ptr);
        }
    }

    void Evaluator::multiply_many(vector<Ciphertext> &encrypteds, const EvaluationKeys &evaluation_keys, Ciphertext &destination, const MemoryPoolHandle &pool)
    {
        // Verify parameters.
        if (encrypteds.size() == 0)
        {
            throw invalid_argument("encrypteds vector must not be empty");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // If there is only one ciphertext, return it after checking validity.
        if (encrypteds.size() == 1)
        {
            // Verify parameters.
            if (encrypteds[0].hash_block_ != parms_.hash_block())
            {
                throw invalid_argument("encrypteds is not valid for encryption parameters");
            }
            destination = encrypteds[0];
            return;
        }

        // Repeatedly multiply and add to the back of the vector until the end is reached
        Ciphertext product(parms_, pool);
        for (size_t i = 0; i < encrypteds.size() - 1; i += 2)
        {
            // We only compare pointers to determine if a faster path can be taken.
            // This is under the assumption that if the two pointers are the same and
            // the parameter sets match, then it makes no sense for one of the ciphertexts
            // to be of different size than the other. More generally, it seems like 
            // a reasonable assumption that if the pointers are the same, then the
            // ciphertexts are the same.
            if (encrypteds[i].data() == encrypteds[i + 1].data())
            {
                square(encrypteds[i], product, pool);
            }
            else
            {
                multiply(encrypteds[i], encrypteds[i + 1], product, pool);
            }
            relinearize(product, evaluation_keys, pool);
            encrypteds.emplace_back(product);
        }
        destination = encrypteds[encrypteds.size() - 1];
    }

    void Evaluator::exponentiate(Ciphertext &encrypted, uint64_t exponent, const EvaluationKeys &evaluation_keys, const MemoryPoolHandle &pool)
    {
        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (exponent == 0)
        {
            throw invalid_argument("exponent cannot be 0");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        if (exponent == 1)
        {
            return;
        }

        // Create a vector of aliased ciphertexts
        vector<Ciphertext> exp_vector(exponent, Ciphertext(parms_, encrypted.size(), encrypted.data()));
        multiply_many(exp_vector, evaluation_keys, encrypted, pool);
    }

    void Evaluator::add_plain(Ciphertext &encrypted, const Plaintext &plain)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (plain.coeff_count() > coeff_count || (plain.coeff_count() == coeff_count && plain[coeff_count - 1] != 0))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#ifdef SEAL_DEBUG
        if (plain.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain.data(), 
            plain.coeff_count(), 1, parms_.plain_modulus().data(), 1))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#endif
        // This is Encryptor::preencrypt
        // Multiply plain by scalar coeff_div_plain_modulus_ and reposition if in upper-half.
        for (int i = 0; i < plain.coeff_count(); i++)
        {
            if (plain[i] >= plain_upper_half_threshold_)
            {
                // Loop over primes
                for (int j = 0; j < coeff_mod_count; j++)
                {
                    uint64_t temp[2]{ 0 };
                    multiply_uint64(*(coeff_div_plain_modulus_.get() + j), plain[i], temp);
                    temp[1] += add_uint64(temp[0], *(upper_half_increment_.get() + j), 0, temp);
                    uint64_t scaled_plain_coeff = barrett_reduce_128(temp, coeff_modulus_[j]);
                    *(encrypted.data() + i + (j * coeff_count)) = add_uint_uint_mod(encrypted[i + (j * coeff_count)], scaled_plain_coeff, coeff_modulus_[j]);
                }
            }
            else
            {
                for (int j = 0; j < coeff_mod_count; j++)
                {
                    uint64_t scaled_plain_coeff = multiply_uint_uint_mod(coeff_div_plain_modulus_[j], plain[i], coeff_modulus_[j]);
                    *(encrypted.data() + i + (j * coeff_count)) = add_uint_uint_mod(encrypted[i + (j * coeff_count)], scaled_plain_coeff, coeff_modulus_[j]);
                }
            }
        }
    }

    void Evaluator::sub_plain(Ciphertext &encrypted, const Plaintext &plain)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (plain.coeff_count() > coeff_count || (plain.coeff_count() == coeff_count && plain[coeff_count - 1] != 0))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#ifdef SEAL_DEBUG
        if (plain.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain.data(), 
            plain.coeff_count(), 1, parms_.plain_modulus().data(), 1))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#endif
        // This is Encryptor::preencrypt changed to subtract instead
        // Multiply plain by scalar coeff_div_plain_modulus_ and reposition if in upper-half.
        for (int i = 0; i < plain.coeff_count(); i++)
        {
            if (plain[i] >= plain_upper_half_threshold_)
            {
                // Loop over primes
                for (int j = 0; j < coeff_mod_count; j++)
                {
                    uint64_t temp[2]{ 0 };
                    multiply_uint64(*(coeff_div_plain_modulus_.get() + j), plain[i], temp);
                    temp[1] += add_uint64(temp[0], *(upper_half_increment_.get() + j), 0, temp);
                    uint64_t scaled_plain_coeff = barrett_reduce_128(temp, coeff_modulus_[j]);
                    *(encrypted.data() + i + (j * coeff_count)) = sub_uint_uint_mod(encrypted[i + (j * coeff_count)], scaled_plain_coeff, coeff_modulus_[j]);
                }
            }
            else
            {
                for (int j = 0; j < coeff_mod_count; j++)
                {
                    uint64_t scaled_plain_coeff = multiply_uint_uint_mod(coeff_div_plain_modulus_[j], plain[i], coeff_modulus_[j]);
                    *(encrypted.data() + i + (j * coeff_count)) = sub_uint_uint_mod(encrypted[i + (j * coeff_count)], scaled_plain_coeff, coeff_modulus_[j]);
                }
            }
        }
    }

    void Evaluator::multiply_plain(Ciphertext &encrypted, const Plaintext &plain, const MemoryPoolHandle &pool)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted_size = encrypted.size();
        int plain_coeff_count = plain.coeff_count();

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
#ifdef SEAL_THROW_ON_MULTIPLY_PLAIN_BY_ZERO
        if (plain.is_zero())
        {
            throw invalid_argument("plain cannot be zero");
        }
#endif
        if (plain.coeff_count() > coeff_count || (plain.coeff_count() == coeff_count && plain[coeff_count - 1] != 0))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#ifdef SEAL_DEBUG
        if (plain.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain.data(), 
            plain.coeff_count(), 1, parms_.plain_modulus().data(), 1))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#endif
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Multiplying just by a constant?
        if (plain_coeff_count == 1)
        {
            if (!qualifiers_.enable_fast_plain_lift)
            {
                Pointer adjusted_coeff(allocate_uint(coeff_mod_count, pool));
                if (plain[0] >= plain_upper_half_threshold_)
                {
                    Pointer decomposed_coeff(allocate_uint(coeff_mod_count, pool));
                    add_uint_uint64(plain_upper_half_increment_.get(), plain[0], coeff_mod_count, adjusted_coeff.get());
                    decompose_single_coeff(adjusted_coeff.get(), decomposed_coeff.get(), pool);

                    for (int i = 0; i < encrypted_size; i++)
                    {
                        for (int j = 0; j < coeff_mod_count; j++)
                        {
                            multiply_poly_scalar_coeffmod(encrypted.data(i) + (j * coeff_count), coeff_count, 
                                decomposed_coeff[j], coeff_modulus_[j], encrypted.data(i) + (j * coeff_count));
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < encrypted_size; i++)
                    {
                        for (int j = 0; j < coeff_mod_count; j++)
                        {
                            multiply_poly_scalar_coeffmod(encrypted.data(i) + (j * coeff_count), coeff_count, 
                                plain[0], coeff_modulus_[j], encrypted.data(i) + (j * coeff_count));
                        }
                    }
                }
                return;
            }
            else
            {
                // Need for lift plain coefficient in RNS form regarding to each qi
                if (plain[0] >= plain_upper_half_threshold_)
                {
                    for (int i = 0; i < encrypted_size; i++)
                    {
                        for (int j = 0; j < coeff_mod_count; j++)
                        {
                            multiply_poly_scalar_coeffmod(encrypted.data(i) + (j * coeff_count), coeff_count, 
                                plain[0] + plain_upper_half_increment_array_[j], coeff_modulus_[j], 
                                encrypted.data(i) + (j * coeff_count));
                        }
                    }
                }
                // No need for lifting
                else
                {
                    for (int i = 0; i < encrypted_size; i++)
                    {
                        for (int j = 0; j < coeff_mod_count; j++)
                        {
                            multiply_poly_scalar_coeffmod(encrypted.data(i) + (j * coeff_count), coeff_count,
                                plain[0], coeff_modulus_[j], encrypted.data(i) + (j * coeff_count));
                        }
                    }
                }
                return;
            }
        }

        // Generic plain case
        Pointer adjusted_poly(allocate_zero_uint(coeff_count * coeff_mod_count, pool));
        Pointer decomposed_poly(allocate_uint(coeff_count * coeff_mod_count, pool));
        uint64_t *poly_to_transform = nullptr;
        if (!qualifiers_.enable_fast_plain_lift)
        {
            // Reposition coefficients.
            const uint64_t *plain_ptr = plain.data();
            uint64_t *adjusted_poly_ptr = adjusted_poly.get();
            for (int i = 0; i < plain_coeff_count; i++, plain_ptr++, adjusted_poly_ptr += coeff_mod_count)
            {
                if (*plain_ptr >= plain_upper_half_threshold_)
                {
                    add_uint_uint64(plain_upper_half_increment_.get(), *plain_ptr, coeff_mod_count,
                        adjusted_poly_ptr);
                }
                else
                {
                    set_uint(*plain_ptr, coeff_mod_count, adjusted_poly_ptr);
                }
            }
            decompose(adjusted_poly.get(), decomposed_poly.get(), pool);
            poly_to_transform = decomposed_poly.get();
        }
        else
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                const uint64_t *plain_ptr = plain.data();
                uint64_t *adjusted_poly_ptr = adjusted_poly.get() + (j * coeff_count);
                uint64_t plain_upper_half_increment = plain_upper_half_increment_array_[j];
                for (int i = 0; i < plain_coeff_count; i++, plain_ptr++, adjusted_poly_ptr++)
                {
                    // Need to lift the coefficient in each qi
                    if (*plain_ptr >= plain_upper_half_threshold_)
                    {
                        *adjusted_poly_ptr = *plain_ptr + plain_upper_half_increment;
                    }
                    // No need for lifting
                    else
                    {
                        *adjusted_poly_ptr = *plain_ptr;
                    }
                }
            }
            poly_to_transform = adjusted_poly.get();
        }

        // Need to multiply each component in encrypted with decomposed_poly (plain poly)
        // Transform plain poly only once
        for (int i = 0; i < coeff_mod_count; i++)
        {
            ntt_negacyclic_harvey(poly_to_transform + (i * coeff_count), coeff_small_ntt_tables_[i]);
        }

        for (int i = 0; i < encrypted_size; i++)
        {
            uint64_t *encrypted_ptr = encrypted.data(i);
            for (int j = 0; j < coeff_mod_count; j++, encrypted_ptr += coeff_count)
            {
                // Explicit inline to avoid unnecessary copy
                //ntt_multiply_poly_nttpoly(encrypted.data(i) + (j * coeff_count), poly_to_transform + (j * coeff_count),
                //    coeff_small_ntt_tables_[j], encrypted.data(i) + (j * coeff_count), pool);

                int coeff_count = coeff_small_ntt_tables_[j].coeff_count() + 1;

                // Lazy reduction
                ntt_negacyclic_harvey_lazy(encrypted_ptr, coeff_small_ntt_tables_[j]);
                dyadic_product_coeffmod(encrypted_ptr, poly_to_transform + (j * coeff_count),
                    coeff_count, coeff_small_ntt_tables_[j].modulus(), encrypted_ptr);
                inverse_ntt_negacyclic_harvey(encrypted_ptr, coeff_small_ntt_tables_[j]);
            }
        }
    }

    void Evaluator::transform_to_ntt(Plaintext &plain, const MemoryPoolHandle &pool)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int plain_coeff_count = plain.coeff_count();

        // Verify parameters.
        if (plain.coeff_count() > coeff_count)
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#ifdef SEAL_DEBUG
        if (plain.significant_coeff_count() >= coeff_count || !are_poly_coefficients_less_than(plain.data(), 
            plain.coeff_count(), 1, parms_.plain_modulus().data(), 1))
        {
            throw invalid_argument("plain is not valid for encryption parameters");
        }
#endif
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        // Resize to fit the entire NTT transformed (ciphertext size) polynomial
        // Note that the new coefficients are automatically set to 0
        plain.resize(coeff_count * coeff_mod_count);

        // Verify if plain lift is needed
        if (!qualifiers_.enable_fast_plain_lift)
        {
            Pointer adjusted_poly(allocate_zero_uint(coeff_count * coeff_mod_count, pool));
            for (int i = 0; i < plain_coeff_count; i++)
            {
                if (plain[i] >= plain_upper_half_threshold_)
                {
                    add_uint_uint64(plain_upper_half_increment_.get(), plain[i], coeff_mod_count,
                        adjusted_poly.get() + (i * coeff_mod_count));
                }
                else
                {
                    set_uint(plain[i], coeff_mod_count, adjusted_poly.get() + (i * coeff_mod_count));
                }
            }
            decompose(adjusted_poly.get(), plain.data(), pool);
        }
        // No need for composed plain lift and decomposition
        else
        {
            for (int j = coeff_mod_count - 1; j >= 0; j--)
            {
                const uint64_t *plain_ptr = plain.data();
                uint64_t *adjusted_poly_ptr = plain.data() + (j * coeff_count);
                uint64_t plain_upper_half_increment = plain_upper_half_increment_array_[j];
                for (int i = 0; i < plain_coeff_count; i++, plain_ptr++, adjusted_poly_ptr++)
                {
                    // Need to lift the coefficient in each qi
                    if (*plain_ptr >= plain_upper_half_threshold_)
                    {
                        *adjusted_poly_ptr = *plain_ptr + plain_upper_half_increment;
                    }
                    // No need for lifting
                    else
                    {
                        *adjusted_poly_ptr = *plain_ptr;
                    }
                }
            }
        }

        // Transform to NTT domain
        for (int i = 0; i < coeff_mod_count; i++)
        {
            ntt_negacyclic_harvey(plain.data() + (i * coeff_count), coeff_small_ntt_tables_[i]);
        }
    }

    void Evaluator::transform_to_ntt(Ciphertext &encrypted)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted_size = encrypted.size();

        // Verify parameters.
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }

        // Transform each polynomial to NTT domain
        for (int i = 0; i < encrypted_size; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                ntt_negacyclic_harvey(encrypted.data(i) + (j * coeff_count), coeff_small_ntt_tables_[j]);
            }
        }
    }

    void Evaluator::transform_from_ntt(Ciphertext &encrypted_ntt)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted_ntt_size = encrypted_ntt.size();

        // Verify parameters.
        if (encrypted_ntt.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted_ntt is not valid for encryption parameters");
        }

        // Transform each polynomial from NTT domain
        for (int i = 0; i < encrypted_ntt_size; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                inverse_ntt_negacyclic_harvey(encrypted_ntt.data(i) + (j * coeff_count), coeff_small_ntt_tables_[j]);
            }
        }
    }

    void Evaluator::multiply_plain_ntt(Ciphertext &encrypted_ntt, const Plaintext &plain_ntt)
    {
        // Extract encryption parameters.
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted_size = encrypted_ntt.size();

        // Verify parameters.
        if (encrypted_ntt.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted_ntt is not valid for encryption parameters");
        }
        if (plain_ntt.coeff_count() != coeff_count * coeff_mod_count)
        {
            throw invalid_argument("plain_ntt is not valid for encryption parameters");
        }
#ifdef SEAL_DEBUG
        for (int i = 0; i < coeff_mod_count; i++)
        {
            if (poly_infty_norm_coeffmod(plain_ntt.data(i * coeff_count), coeff_count, 
                coeff_modulus_[i]) >= coeff_modulus_[i].value())
            {
                throw invalid_argument("plain_ntt is not valid for encryption parameters");
            }
            if (plain_ntt[coeff_count - 1 + (i * coeff_count)] != 0)
            {
                throw invalid_argument("plain_ntt is not valid for encryption parameters");
            }
        }
#endif
#ifdef SEAL_THROW_ON_MULTIPLY_PLAIN_BY_ZERO
        if (plain_ntt.is_zero())
        {
            throw invalid_argument("plain_ntt cannot be zero");
        }
#endif
        for (int i = 0; i < encrypted_size; i++)
        {
            for (int j = 0; j < coeff_mod_count; j++)
            {
                dyadic_product_coeffmod(encrypted_ntt.data(i) + (j * coeff_count), plain_ntt.data() + (j * coeff_count),
                    coeff_count - 1, coeff_modulus_[j], encrypted_ntt.data(i) + (j * coeff_count));
            }
        }
    }

    void Evaluator::apply_galois(Ciphertext &encrypted, uint64_t galois_elt, const GaloisKeys &galois_keys, const MemoryPoolHandle &pool)
    {
        // Extract paramters
        int coeff_count = parms_.poly_modulus().coeff_count();
        int coeff_mod_count = static_cast<int>(coeff_modulus_.size());
        int encrypted_size = encrypted.size();

        // Verify parameters
        if (!(galois_elt & 1) || (galois_elt >= 2 * static_cast<uint64_t>(coeff_count) - 1))
        {
            throw invalid_argument("galois element is not valid");
        }
        if (encrypted.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("encrypted is not valid for encryption parameters");
        }
        if (galois_keys.hash_block_ != parms_.hash_block())
        {
            throw invalid_argument("galois_keys is not valid for encryption parameters");
        }
        if (encrypted_size > 2)
        {
            throw invalid_argument("ciphertext size must be 2");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        int n = coeff_count - 1;
        int m = n << 1;
        int subgroup_size = n >> 1;
        int n_power_of_two = get_power_of_two(n);

        // Check if Galois key is generated or not.
        // If not, attempt a bit decomposition; maybe we have log(n) many keys
        if (!galois_keys.has_key(galois_elt))
        {
            // galois_elt = 3^order1 * (-1)^order2
            uint64_t order1 = Zmstar_to_generator_.at(galois_elt).first;
            uint64_t order2 = Zmstar_to_generator_.at(galois_elt).second;

            // We use either 3 or -3 as our generator, depending on which gives smaller HW
            uint64_t two_power_of_gen = 3;

            // Does order1 or n/2-order1 have smaller Hamming weight?
            if (hamming_weight(subgroup_size - order1) < hamming_weight(order1))
            {
                order1 = subgroup_size - order1;
                try_mod_inverse(3, m, two_power_of_gen);
            }

            while(order1)
            {
                if (order1 & 1)
                {
                    if (!galois_keys.has_key(two_power_of_gen))
                    {
                        throw invalid_argument("galois key not present");
                    }
                    apply_galois(encrypted, two_power_of_gen, galois_keys, pool);
                }
                two_power_of_gen *= two_power_of_gen;
                two_power_of_gen &= (m - 1);
                order1 >>= 1;
            }
            if (order2)
            {
                if (!galois_keys.has_key(m - 1))
                {
                    throw invalid_argument("galois key not present");
                }
                apply_galois(encrypted, m - 1, galois_keys, pool);
            }
            return;
        }

        // Apply Galois for each ciphertext
        Pointer temp0(allocate_zero_uint(coeff_count * coeff_mod_count, pool));
        for (int i = 0; i < coeff_mod_count; i++)
        {
            util::apply_galois(encrypted.data() + (i * coeff_count), n_power_of_two,
                galois_elt, coeff_modulus_[i], temp0.get() + (i * coeff_count));
        }
        Pointer temp1(allocate_zero_uint(coeff_count * coeff_mod_count, pool));
        for (int i = 0; i < coeff_mod_count; i++)
        {
            util::apply_galois(encrypted.data(1) + (i * coeff_count), n_power_of_two,
                galois_elt, coeff_modulus_[i], temp1.get() + (i * coeff_count));
        }

        // Calculate (temp1 * galois_key.first, temp1 * galois_key.second) + (temp0, 0)
        const uint64_t *encrypted_coeff = temp1.get();
        Pointer encrypted_coeff_prod_inv_coeff(allocate_uint(coeff_count, pool));

        // decompose encrypted_array[count-1] into base w
        // want to create an array of polys, each of whose components i is (encrypted_array[count-1])^(i) - in the notation of FV paper
        // This allocation stores one of the decomposed factors modulo one of the primes
        Pointer decomp_encrypted_last(allocate_uint(coeff_count, pool));

        // Lazy reduction
        Pointer wide_innerresult0(allocate_zero_poly(coeff_count, 2 * coeff_mod_count, pool));
        Pointer wide_innerresult1(allocate_zero_poly(coeff_count, 2 * coeff_mod_count, pool));
        Pointer innerresult(allocate_poly(coeff_count, coeff_mod_count, pool));
        Pointer temp_decomp_coeff(allocate_uint(coeff_count, pool));

        /*
        For lazy reduction to work here, we need to ensure that the 128-bit accumulators (wide_innerresult0 and wide_innerresult1)
        do not overflow. Since the modulus primes are at most 60 bits, if the total number of summands is K, then the size of the
        total sum of products (without reduction) is at most 62 + 60 + bit_length(K). We need this to be at most 128, thus we need
        bit_length(K) <= 6. Thus, we need K <= 63. In this case, this means sum_i galois_keys.key(galois_elt)[i].size() / 2 <= 63.
        */
        for (int i = 0; i < coeff_mod_count; i++)
        {
            multiply_poly_scalar_coeffmod(encrypted_coeff + (i * coeff_count), coeff_count, 
                inv_coeff_products_mod_coeff_array_[i], coeff_modulus_[i], encrypted_coeff_prod_inv_coeff.get());

            int shift = 0;
            const Ciphertext &key_component_ref = galois_keys.key(galois_elt)[i];
            int keys_size = key_component_ref.size();
            for (int k = 0; k < keys_size; k += 2)
            {
                const uint64_t *key_ptr_0 = key_component_ref.data(k);
                const uint64_t *key_ptr_1 = key_component_ref.data(k + 1);

                // Decompose here
                int decomposition_bit_count = galois_keys.decomposition_bit_count();
                for (int coeff_index = 0; coeff_index < coeff_count; coeff_index++)
                {
                    decomp_encrypted_last[coeff_index] = encrypted_coeff_prod_inv_coeff[coeff_index] >> shift;
                    decomp_encrypted_last[coeff_index] &= (1ULL << decomposition_bit_count) - 1;
                }

                uint64_t *wide_innerresult0_ptr = wide_innerresult0.get();
                uint64_t *wide_innerresult1_ptr = wide_innerresult1.get();
                for (int j = 0; j < coeff_mod_count; j++)
                {
                    uint64_t *temp_decomp_coeff_ptr = temp_decomp_coeff.get();
                    set_uint_uint(decomp_encrypted_last.get(), coeff_count, temp_decomp_coeff_ptr);

                    // We don't reduce here, so might get up to two extra bits. Thus 62 bits at most.
                    ntt_negacyclic_harvey_lazy(temp_decomp_coeff_ptr, coeff_small_ntt_tables_[j]);

                    // Lazy reduction
                    uint64_t wide_innerproduct[2];
                    for (int m = 0; m < coeff_count; m++, wide_innerresult0_ptr += 2)
                    {
                        multiply_uint64(*temp_decomp_coeff_ptr++, *key_ptr_0++, wide_innerproduct);
                        unsigned char carry = add_uint64(wide_innerresult0_ptr[0], wide_innerproduct[0], 0,
                            wide_innerresult0_ptr);
                        wide_innerresult0_ptr[1] += wide_innerproduct[1] + carry;
                    }

                    temp_decomp_coeff_ptr = temp_decomp_coeff.get();
                    for (int m = 0; m < coeff_count; m++, wide_innerresult1_ptr += 2)
                    {
                        multiply_uint64(*temp_decomp_coeff_ptr++, *key_ptr_1++, wide_innerproduct);
                        unsigned char carry = add_uint64(wide_innerresult1_ptr[0], wide_innerproduct[0], 0,
                            wide_innerresult1_ptr);
                        wide_innerresult1_ptr[1] += wide_innerproduct[1] + carry;
                    }
                }
                shift += decomposition_bit_count;
            }
        }

        uint64_t *temp_ptr = temp0.get();
        uint64_t *innerresult_poly_ptr = innerresult.get();
        uint64_t *wide_innerresult_poly_ptr = wide_innerresult0.get();
        uint64_t *encrypted_ptr = encrypted.data();
        uint64_t *innerresult_coeff_ptr = innerresult_poly_ptr;
        uint64_t *wide_innerresult_coeff_ptr = wide_innerresult_poly_ptr;
        for (int i = 0; i < coeff_mod_count; i++, innerresult_poly_ptr += coeff_count,
            wide_innerresult_poly_ptr += 2 * coeff_count, encrypted_ptr += coeff_count,
            temp_ptr += coeff_count)
        {
            for (int m = 0; m < coeff_count; m++, wide_innerresult_coeff_ptr += 2)
            {
                *innerresult_coeff_ptr++ = barrett_reduce_128(wide_innerresult_coeff_ptr, coeff_modulus_[i]);
            }
            inverse_ntt_negacyclic_harvey(innerresult_poly_ptr, coeff_small_ntt_tables_[i]);
            add_poly_poly_coeffmod(temp_ptr, innerresult_poly_ptr, coeff_count,
                coeff_modulus_[i], encrypted_ptr);
        }

        innerresult_poly_ptr = innerresult.get();
        wide_innerresult_poly_ptr = wide_innerresult1.get();
        encrypted_ptr = encrypted.data(1);
        wide_innerresult_coeff_ptr = wide_innerresult_poly_ptr;
        for (int i = 0; i < coeff_mod_count; i++, innerresult_poly_ptr += coeff_count,
            wide_innerresult_poly_ptr += 2 * coeff_count, encrypted_ptr += coeff_count)
        {
            innerresult_coeff_ptr = encrypted_ptr;
            for (int m = 0; m < coeff_count; m++, wide_innerresult_coeff_ptr += 2)
            {
                *innerresult_coeff_ptr++ = barrett_reduce_128(wide_innerresult_coeff_ptr, coeff_modulus_[i]);
            }
            inverse_ntt_negacyclic_harvey(encrypted_ptr, coeff_small_ntt_tables_[i]);
        }
    }

    void Evaluator::rotate_rows(Ciphertext &encrypted, int steps, const GaloisKeys &galois_keys, const MemoryPoolHandle &pool)
    {
        if (!qualifiers_.enable_batching)
        {
            throw logic_error("encryption parameters do not support batching");
        }

        // Is there anything to do?
        if (steps == 0)
        {
            return;
        }

        // Extract sign of steps. When steps is positive, the rotation is to the left,
        // and when steps is negative, it is to the right.
        bool sign = steps < 0;
        uint32_t pos_steps = abs(steps);
        uint32_t n = parms_.poly_modulus().coeff_count() - 1;
        uint32_t m_power_of_two = get_power_of_two(n) + 1;

        if (pos_steps >= (n >> 1))
        {
            throw invalid_argument("step count too large");
        }

        pos_steps &= (1UL << m_power_of_two) - 1;
        if (sign)
        {
            steps = (n >> 1) - pos_steps;
        }
        else
        {
            steps = pos_steps;
        }

        // Construct Galois element for row rotation
        int gen = 3;
        uint64_t galois_elt = 1;
        for (int i = 0; i < steps; i++)
        {
            galois_elt *= gen;
            galois_elt &= (1ULL << m_power_of_two) - 1;
        }

        // Perform rotation and key switching
        apply_galois(encrypted, galois_elt, galois_keys, pool);
    }

    //void Evaluator::polynomial_evaluation(const Ciphertext &encrypted, vector<uint64_t> &coeff, const EvaluationKeys &evaluation_keys, const SmallModulus &eval_plain_modulus, Ciphertext &destination)
    //{
    //    // Save original parameters for recover at the end of this algorithm
    //    EncryptionParameters origin_parms_(parms_);

    //    // Temporarily change the plain_modulus to eval_plain_modulus
    //    parms_.set_plain_modulus(eval_plain_modulus);
    //    Ciphertext encrypted_copy(encrypted);
    //    encrypted_copy.hash_block_ = parms_.hash_block();

    //    int degree = coeff.size() - 1;

    //    // Compute parameter k = sqrt(degree/2), m = smallest interger with degree_prime = k * (2^m - 1) > degree
    //    int k = optimal_parameter_paterson(degree);
    //    int degree_prime;
    //    int m = 1;
    //    while (1)
    //    {
    //        degree_prime = (exponentiate_uint64(2, m) - 1) * k;
    //        if (degree_prime > degree)
    //        {
    //            break;
    //        }
    //        else
    //        {
    //            m++;
    //        }
    //    }

    //    // If degree is low just compute all power and dot product
    //    int number_of_nonscalar_mult = (k - 1) + 2 * (m - 1) + (1 << (m - 1)) - 1;
    //    if (degree < number_of_nonscalar_mult || degree < 3) 
    //    {
    //        vector<Ciphertext> power_of_all(degree + 1);
    //        compute_all_powers(encrypted_copy, degree, evaluation_keys, power_of_all);
    //        polynomial_evaluation(power_of_all, coeff, evaluation_keys, destination);

    //        // Go back to original parms_
    //        parms_ = origin_parms_;
    //        destination.hash_block_ = parms_.hash_block();
    //        return;
    //    }

    //    // Compute encryption of baby step (1, msg, msg^2, ... , msg^k)
    //    vector<Ciphertext> baby_step(k + 1);
    //    compute_all_powers(encrypted_copy, k, evaluation_keys, baby_step);

    //    // Compute encryption of giant step (msg^(2k), ... , msg^(2^{m-1}k))
    //    vector<Ciphertext> giant_step(m - 1);
    //    square(baby_step[k], giant_step[0]);
    //    relinearize(giant_step[0], evaluation_keys);
    //    for (int i = 0; i < m - 2; i++)
    //    {
    //        square(giant_step[i], giant_step[i + 1]);
    //        relinearize(giant_step[i + 1], evaluation_keys);
    //    }

    //    // Set f'(x) = X^degree_prime + f(x)
    //    vector<uint64_t> f_prime(degree_prime + 1);
    //    for (int i = 0; i < degree_prime + 1; i++)
    //    {
    //        if (i < coeff.size())
    //        {
    //            f_prime[i] = coeff[i];
    //        }
    //        else if (i < degree_prime)
    //        {
    //            f_prime[i] = 0;
    //        }
    //    }
    //    f_prime[degree_prime] = 1;

    //    //Evaluate f_prime
    //    Ciphertext encrypted_f_prime(parms_);
    //    paterson_stockmeyer(baby_step, giant_step, encrypted_copy, f_prime, evaluation_keys, encrypted_f_prime, k, m);

    //    //Compute encryption of X^degree_prime = X^k * X^2k ... X^{2^{m-1}k}
    //    Ciphertext power_of_degree_prime(parms_);
    //    multiply(baby_step[k], giant_step[0], power_of_degree_prime);
    //    relinearize(power_of_degree_prime, evaluation_keys);
    //    for (int i = 0; i < m - 2; i++)
    //    {
    //        multiply(power_of_degree_prime, giant_step[i + 1], power_of_degree_prime);
    //        relinearize(power_of_degree_prime, evaluation_keys);
    //    }

    //    //Substract X^degree_prime from temp
    //    sub(encrypted_f_prime, power_of_degree_prime, destination);

    //    // Go back to original parms_
    //    parms_ = origin_parms_;
    //    destination.hash_block_ = parms_.hash_block();
    //}

    //void Evaluator::digit_extraction(Ciphertext &encrypted, const EvaluationKeys &evaluation_keys, vector<Ciphertext> &destination, long p, long e, long r)
    //{
    //    if (e < 2)
    //    {
    //        throw invalid_argument("e must be at least 2");
    //    }

    //    //Extract encryption parameters.
    //    int coeff_count = parms_.poly_modulus().coeff_count();
    //    int coeff_mod_count = parms_.coeff_modulus().size();
    //    int encrypted_size = encrypted.size();

    //    //Find k such that p^k > (e-1)(p-1)+1
    //    long k = 1;
    //    long pow_of_p = 1;
    //    while (1)
    //    {
    //        pow_of_p *= p;
    //        if (pow_of_p < (e - 1)*(p - 1) + 1)
    //        {
    //            k++;
    //        }
    //        else
    //        {
    //            break;
    //        }
    //    }

    //    //Initialize destination
    //    destination.resize(r);
    //    for (int i = 0; i < r; i++)
    //    {
    //        destination[i].resize(parms_, encrypted_size);
    //    }

    //    //lift[i] = poly_lift(3, i+1) which makes x + O(3^{i+1}) to x + O(3^{i+2})
    //    vector<vector<uint64_t> > lift(r - 1);
    //    for (int i = 0; i < r - 1; i++) 
    //    {
    //        lift[i].resize(p + 1);
    //        lift[i] = compute_lift_uint64(p, i + 1);
    //    }

    //    //remainlsd[i](x) = [x]_3 in mod 3^{i+2}
    //    vector<vector<uint64_t> > remainlsd;
    //    remainlsd.resize(e - 1);
    //    for (int i = 0; i < e - 1; i++) 
    //    {
    //        int degree = (i + 1)*(p - 1) + 1;
    //        remainlsd[i].resize(degree + 1);
    //        if (i >= (e - r - 1)) remainlsd[i] = compute_remainlsd_uint64(p, i + 2);
    //    }

    //    //Temp ciphertexts for computation
    //    vector<vector<Ciphertext> > lift_encrypted;
    //    lift_encrypted.resize(r);
    //    for (int i = 0; i < r; i++)
    //    {
    //        lift_encrypted[i].resize(k - 1);
    //    }

    //    Ciphertext cipher(parms_, pool_);
    //    cipher = encrypted;

    //    uint64_t plain_modulus_ = parms_.plain_modulus().value();
    //    for (int i = 0; i < r; i++)
    //    {
    //        SmallModulus eval_plain_modulus(plain_modulus_);
    //        if (i < r - 1)
    //        {
    //            polynomial_evaluation(cipher, lift[0], evaluation_keys, eval_plain_modulus, lift_encrypted[i][0]);
    //        }
    //        polynomial_evaluation(cipher, remainlsd[e - i - 2], evaluation_keys, eval_plain_modulus, destination[i]);
    //        for (int j = 0; j < k - 2; j++)
    //        {
    //            if (i + j < r - 1)
    //            {
    //                polynomial_evaluation(lift_encrypted[i][j], lift[j + 1], evaluation_keys, eval_plain_modulus, 
    //                    lift_encrypted[i][j + 1]);
    //            }
    //        }
    //        cipher = encrypted;
    //        if (i < r - 1)
    //        {
    //            for (int j = 0; j < i + 1; j++)
    //            {
    //                if (i - j > k - 2)
    //                {
    //                    sub(cipher, destination[j], cipher);
    //                }
    //                else
    //                {
    //                    sub(cipher, lift_encrypted[j][i - j], cipher);
    //                }
    //            }
    //        }
    //        plain_modulus_ /= p;
    //    }
    //}

    ///**
    //Recursive polynomial evaution methdod,
    //First, f(x) = ((X^{k2^{step-1}}) + a(x)) * q(x) + X^(k2^{step-1}-1) + b(x)
    //Second, Compute q(x) and X^(k2^{step-1}-1) + b(x) using recursive function call
    //Third, the step = 1, just do dot product and return
    //*/
    //void Evaluator::paterson_stockmeyer(vector<Ciphertext> &baby_step, vector<Ciphertext> &giant_step, Ciphertext &encrypted, vector<uint64_t> &coeff, const EvaluationKeys &evaluation_keys, Ciphertext &destination, int k, int step)
    //{
    //    //Check input coeff degree = k* (2^step - 1)?
    //    int current_degree = (exponentiate_uint64(2, step) - 1) * k;
    //    if (coeff.size() != current_degree + 1)
    //    {
    //        invalid_argument("degree is incorrect for PatersonStockmeyer method");
    //    }

    //    //If the degree is k, do just dot product
    //    if (step == 1) 
    //    {
    //        polynomial_evaluation(baby_step, coeff, evaluation_keys, destination);
    //        return;
    //    }

    //    uint64_t plain_modulus_ = parms_.plain_modulus().value();

    //    // Set q(x) and r(x) to be a f(x) = q(x) * X^{k2^{step-1}} + r(x)
    //    uint64_t temp_degree = (exponentiate_uint64(2, step - 1) * k) - 1;
    //    uint64_t next_degree = (exponentiate_uint64(2, step - 1) - 1) * k;
    //    vector<uint64_t> qx(next_degree + 1);
    //    vector<uint64_t> rx(temp_degree + 1);
    //    for (uint64_t i = 0; i < coeff.size(); i++)
    //    {
    //        if (i < temp_degree + 1)
    //        {
    //            rx[i] = coeff[i];
    //        }
    //        else
    //        {
    //            qx[i - temp_degree - 1] = coeff[i];
    //        }
    //    }

    //    // r(x) = r(x) - X^(k2^{step-1}-1)
    //    if (rx[next_degree] > 0)
    //    {
    //        rx[next_degree] -= 1;
    //    }
    //    else
    //    {
    //        rx[next_degree] += plain_modulus_ - 1;
    //    }

    //    // convert from vector to BigPoly
    //    BigPoly poly_qx(rx.size(), 64);
    //    set_uint_uint(qx.data(), qx.size(), poly_qx.data());

    //    for (int i = qx.size(); i < rx.size(); i++)
    //    {
    //        poly_qx[i].set_zero();
    //    }

    //    // convert from vector to BigPoly
    //    BigPoly poly_rx(rx.size(), 64);
    //    set_uint_uint(rx.data(), rx.size(), poly_rx.data());

    //    // r(x) = a(x) * q(x) + b(x)
    //    BigPoly poly_ax(poly_rx.coeff_count(), parms_.plain_modulus().bit_count());
    //    BigPoly poly_bx(poly_rx.coeff_count(), parms_.plain_modulus().bit_count());
    //    divide_poly_poly_coeffmod(poly_rx.data(), poly_qx.data(), poly_rx.coeff_count(), parms_.plain_modulus(), poly_ax.data(), poly_bx.data(), pool_);

    //    //Convert bigpoly to vector<uint64_t>
    //    vector<uint64_t> ax(poly_ax.significant_coeff_count());
    //    for (int i = 0; i < poly_ax.significant_coeff_count(); i++)
    //    {
    //        ax[i] = *(poly_ax[i].data());
    //    }

    //    // Evaluate ax using baby_step (dot product)
    //    Ciphertext encrypted_ax(parms_, pool_);
    //    polynomial_evaluation(baby_step, ax, evaluation_keys, encrypted_ax);

    //    // Compute (X^{k2^{step-1}}) + a(x) using addition
    //    Ciphertext encrypted_ax_add_power_of_two(encrypted_ax);
    //    add(giant_step[step - 2], encrypted_ax_add_power_of_two);

    //    // Compute bx polynomial = X^(k2^{step-1}-1) + b(x)
    //    vector<uint64_t> bx(next_degree + 1);
    //    for (int i = 0; i < poly_bx.significant_coeff_count(); i++)
    //    {
    //        bx[i] = *(poly_bx[i].data());
    //    }
    //    for (int i = poly_bx.significant_coeff_count(); i < next_degree; i++)
    //    {
    //        bx[i] = 0;
    //    }
    //    bx[next_degree] = 1;

    //    // Evaluate q(x) recursively
    //    Ciphertext encrypted_qx(parms_, pool_);
    //    paterson_stockmeyer(baby_step, giant_step, encrypted, qx, evaluation_keys, encrypted_qx, k, step - 1);

    //    // Evaluate X^(k2^{step-1}-1) + b(x) recursively
    //    Ciphertext encrypted_bx(parms_, pool_);
    //    paterson_stockmeyer(baby_step, giant_step, encrypted, bx, evaluation_keys, encrypted_bx, k, step - 1);

    //    // Compute ((X^{k2^{step-1}}) + a(x)) * q(x)
    //    Ciphertext encrypted_part(parms_, pool_);
    //    multiply(encrypted_ax_add_power_of_two, encrypted_qx, encrypted_part);
    //    relinearize(encrypted_part, evaluation_keys);

    //    //Compute f(x) = [((X^{k2^{step-1}}) + a(x)) * q(x)] + X^(k2^{step-1}-1) + b(x)
    //    destination = encrypted_part;
    //    add(encrypted_bx, destination);
    //}

    //void Evaluator::polynomial_evaluation(const vector<Ciphertext> &all_powers_encrypted, vector<uint64_t> &coeff, const EvaluationKeys &evaluation_keys, Ciphertext &destination)
    //{
    //    // Check whether we have enough all powers
    //    if (all_powers_encrypted.size() < coeff.size())
    //    {
    //        throw invalid_argument("there are not enough powers_encrypted");
    //    }

    //    // Extract encryption parameters.
    //    int coeff_count = parms_.poly_modulus().coeff_count();
    //    int coeff_mod_count = parms_.coeff_modulus().size();
    //    int encrypted_size = all_powers_encrypted[0].size();

    //    // Initialize destination
    //    destination.resize(parms_, encrypted_size);
    //    destination.set_zero();

    //    // Dot product
    //    int degree = coeff.size() - 1;

    //    for (int i = 0; i < degree + 1; i++)
    //    {
    //        if (coeff[i] != 0)
    //        {
    //            for (int j = 0; j < encrypted_size; j++)
    //            {
    //                const uint64_t *all_powers_encrypted_ptr = all_powers_encrypted[i].data(j);
    //                uint64_t *destination_ptr = destination.data(j);
    //                for (int k = 0; k < coeff_mod_count; k++)
    //                {
    //                    for (int m = 0; m < coeff_count; m++)
    //                    {
    //                        // Lazy reduction
    //                        uint64_t temp[2];
    //                        multiply_uint64(all_powers_encrypted_ptr[m + (k * coeff_count)], coeff[i], temp);
    //                        temp[1] += add_uint64(destination_ptr[m + (k * coeff_count)], temp[0], 0, temp);
    //                        destination_ptr[m + (k * coeff_count)] = barrett_reduce_128(temp, coeff_modulus_[k]);
    //                    }
    //                }
    //            }
    //        }
    //    }
    //}

    //void Evaluator::compute_all_powers(const Ciphertext &encrypted, int degree, const EvaluationKeys &evaluation_keys, vector<Ciphertext> &destination)
    //{
    //    // Verify parameters.
    //    if (encrypted.hash_block_ != parms_.hash_block())
    //    {
    //        throw invalid_argument("encrypted is not valid for encryption parameters");
    //    }

    //    // Extract encryption parameters.
    //    int coeff_count = parms_.poly_modulus().coeff_count();
    //    int coeff_mod_count = parms_.coeff_modulus().size();
    //    int coeff_bit_count = coeff_mod_count * bits_per_uint64;
    //    int encrypted_size = encrypted.size();

    //    // Resize destination and clear
    //    destination.resize(degree + 1);
    //    destination[0].resize(parms_, encrypted_size);
    //    destination[0].set_zero();

    //    // Set destination[0] = (Delta, 0, ..., 0)
    //    BigUInt coeff_div_current_plain_modulus_(coeff_bit_count);
    //    Pointer temp(allocate_uint(coeff_mod_count, pool_));
    //    Pointer wide_plain_modulus(allocate_uint(coeff_mod_count, pool_));
    //    set_uint_uint(parms_.plain_modulus().data(), parms_.plain_modulus().uint64_count(), coeff_mod_count, 
    //        wide_plain_modulus.get());
    //    divide_uint_uint(product_modulus_.get(), wide_plain_modulus.get(), coeff_mod_count, 
    //        coeff_div_current_plain_modulus_.data(), temp.get(), pool_);
    //    set_uint_uint(coeff_div_current_plain_modulus_.data(), coeff_mod_count, destination[0].data());
    //    decompose(destination[0].data());

    //    //Compute all X^{2^i} using square function
    //    int log_2_degree = static_cast<int>(floor(log2(degree)));
    //    vector<Ciphertext> all_two_powers_encrypted(log_2_degree + 1);
    //    all_two_powers_encrypted[0] = encrypted;
    //    for (int i = 1; i <= log_2_degree; i++)
    //    {
    //        square(all_two_powers_encrypted[i - 1], all_two_powers_encrypted[i]);
    //        relinearize(all_two_powers_encrypted[i], evaluation_keys);
    //    }

    //    // Compute all X^{i} using multiplication tree
    //    for (uint64_t i = 1; i <= degree; i++)
    //    {
    //        uint64_t i1 = hamming_weight_split(i);
    //        uint64_t i2 = i - i1;
    //        if (i1 == 0 || i2 == 0)
    //        {
    //            destination[i].resize(parms_, encrypted_size);
    //            destination[i] = all_two_powers_encrypted[get_significant_bit_count(static_cast<uint64_t>(i)) - 1];
    //        }
    //        else
    //        {
    //            destination[i].resize(parms_, encrypted_size);
    //            multiply(destination[i1], destination[i2], destination[i]);
    //            relinearize(destination[i], evaluation_keys);
    //        }
    //    }
    //}
}
