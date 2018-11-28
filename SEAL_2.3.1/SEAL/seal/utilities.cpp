#include <cmath>
#include "seal/utilities.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintarithmod.h"
#include "seal/util/polyarithmod.h"
#include "seal/util/polycore.h"
#include "seal/util/polyarith.h"
#include "seal/util/modulus.h"
#include "seal/util/polymodulus.h"
#include "seal/decryptor.h"
#include "seal/simulator.h"

using namespace std;
using namespace seal::util;

namespace seal
{
    namespace
    {
        ConstPointer duplicate_uint_if_needed(const BigUInt &uint, int new_uint64_count, bool force, MemoryPool &pool)
        {
            return duplicate_uint_if_needed(uint.data(), uint.uint64_count(), new_uint64_count, force, pool);
        }

        ConstPointer duplicate_poly_if_needed(const BigPoly &poly, int new_coeff_count, int new_coeff_uint64_count, bool force, MemoryPool &pool)
        {
            return duplicate_poly_if_needed(poly.data(), poly.coeff_count(), poly.coeff_uint64_count(), new_coeff_count, new_coeff_uint64_count, force, pool);
        }

        bool are_poly_coefficients_less_than(const BigPoly &poly, const BigUInt &max_coeff)
        {
            return util::are_poly_coefficients_less_than(poly.data(), poly.coeff_count(), poly.coeff_uint64_count(), max_coeff.data(), max_coeff.uint64_count());
        }
    }

    BigUInt poly_infty_norm(const BigPoly &poly)
    {
        if (poly.is_zero())
        {
            return BigUInt();
        }

        int coeff_count = poly.coeff_count();
        int coeff_bit_count = poly.coeff_bit_count();
        int coeff_uint64_count = divide_round_up(coeff_bit_count, bits_per_uint64);

        BigUInt result(coeff_bit_count);
        util::poly_infty_norm(poly.data(), coeff_count, coeff_uint64_count, result.data());

        return result;
    }

    BigUInt poly_infty_norm_coeffmod(const BigPoly &poly, const BigUInt &modulus, const MemoryPoolHandle &pool)
    {
        if (modulus.is_zero())
        {
            throw invalid_argument("modulus cannot be zero");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        if (poly.is_zero())
        {
            return BigUInt();
        }

        int poly_coeff_count = poly.coeff_count();
        int poly_coeff_bit_count = poly.coeff_bit_count();
        int poly_coeff_uint64_count = divide_round_up(poly_coeff_bit_count, bits_per_uint64);

        Modulus mod(modulus.data(), modulus.uint64_count(), pool);
        BigUInt result(modulus.significant_bit_count());
        util::poly_infty_norm_coeffmod(poly.data(), poly_coeff_count, poly_coeff_uint64_count, mod, result.data(), pool);

        return result;
    }

    void exponentiate_uint_mod(const BigUInt &operand, const BigUInt &exponent, 
        const BigUInt &modulus, BigUInt &destination, const MemoryPoolHandle &pool)
    {
        if (operand.significant_bit_count() > modulus.significant_bit_count())
        {
            throw invalid_argument("operand is not reduced");
        }
        if (operand.is_zero() && exponent == 0)
        {
            throw invalid_argument("undefined operation");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        if (operand.is_zero())
        {
            destination.set_zero();
            return;
        }

        if (destination.bit_count() != modulus.significant_bit_count())
        {
            destination.resize(modulus.significant_bit_count());
        }

        ConstPointer operand_ptr = duplicate_uint_if_needed(operand, modulus.uint64_count(), false, pool);
        util::exponentiate_uint_mod(operand_ptr.get(), exponent.data(), exponent.uint64_count(), Modulus(modulus.data(), modulus.uint64_count(), pool), destination.data(), pool);
    }

    void exponentiate_poly_polymod_coeffmod(const BigPoly &operand, const BigUInt &exponent, 
        const BigPoly &poly_modulus, const BigUInt &coeff_modulus, BigPoly &destination, const MemoryPoolHandle &pool)
    {
        if (operand.significant_coeff_count() > poly_modulus.coeff_count() ||
            operand.significant_coeff_bit_count() > coeff_modulus.significant_bit_count())
        {
            throw invalid_argument("operand is not reduced");
        }
        if (exponent < 0)
        {
            throw invalid_argument("exponent must be a non-negative integer");
        }
        if (operand.is_zero() && exponent == 0)
        {
            throw invalid_argument("undefined operation");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        if (operand.is_zero())
        {
            destination.set_zero();
            return;
        }

        if (destination.coeff_bit_count() != coeff_modulus.significant_bit_count() || 
            destination.coeff_count() != poly_modulus.coeff_count())
        {
            destination.resize(poly_modulus.coeff_count(), coeff_modulus.significant_bit_count());
        }

        ConstPointer operand_ptr = duplicate_poly_if_needed(operand, poly_modulus.coeff_count(), coeff_modulus.uint64_count(), false, pool);
        util::exponentiate_poly_polymod_coeffmod(operand_ptr.get(), exponent.data(), exponent.uint64_count(),
            PolyModulus(poly_modulus.data(), poly_modulus.coeff_count(), poly_modulus.coeff_uint64_count()), 
            Modulus(coeff_modulus.data(), coeff_modulus.uint64_count(), pool), 
            destination.data(), pool);
    }

    void poly_eval_poly(const BigPoly &poly_to_evaluate, const BigPoly &poly_to_evaluate_at, 
        BigPoly &destination, const MemoryPoolHandle &pool)
    {
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        int poly_to_eval_coeff_uint64_count = divide_round_up(poly_to_evaluate.coeff_bit_count(), bits_per_uint64);
        int value_coeff_uint64_count = divide_round_up(poly_to_evaluate_at.coeff_bit_count(), bits_per_uint64);

        if (poly_to_evaluate.is_zero())
        {
            destination.set_zero();
            return;
        }
        if (poly_to_evaluate_at.is_zero())
        {
            destination.resize(1, poly_to_evaluate.coeff_bit_count());
            set_uint_uint(poly_to_evaluate.data(), poly_to_eval_coeff_uint64_count, destination.data());
            return;
        }

        int result_coeff_count = (poly_to_evaluate.significant_coeff_count() - 1) * (poly_to_evaluate_at.significant_coeff_count() - 1) + 1;
        int result_coeff_bit_count = poly_to_evaluate.coeff_bit_count() + (poly_to_evaluate.coeff_count() - 1) * poly_to_evaluate_at.coeff_bit_count();
        int result_coeff_uint64_count = divide_round_up(result_coeff_bit_count, bits_per_uint64);
        destination.resize(result_coeff_count, result_coeff_bit_count);

        util::poly_eval_poly(poly_to_evaluate.data(), poly_to_evaluate.coeff_count(), poly_to_eval_coeff_uint64_count, 
            poly_to_evaluate_at.data(), poly_to_evaluate_at.coeff_count(), value_coeff_uint64_count, 
            result_coeff_count, result_coeff_uint64_count, destination.data(), pool);
    }

    BigPoly poly_eval_poly(const BigPoly &poly_to_evaluate, const BigPoly &poly_to_evaluate_at, const MemoryPoolHandle &pool)
    {
        BigPoly result;
        poly_eval_poly(poly_to_evaluate, poly_to_evaluate_at, result, pool);
        return result;
    }

    void poly_eval_poly_polymod_coeffmod(const BigPoly &poly_to_evaluate, const BigPoly &poly_to_evaluate_at, 
        const BigPoly &poly_modulus, const BigUInt &coeff_modulus, BigPoly &destination, const MemoryPoolHandle &pool)
    {
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }
        if (poly_to_evaluate.significant_coeff_count() > poly_modulus.coeff_count() ||
            poly_to_evaluate.significant_coeff_bit_count() > coeff_modulus.significant_bit_count())
        {
            throw invalid_argument("poly_to_evaluate is not reduced");
        }
        if (poly_to_evaluate_at.significant_coeff_count() > poly_modulus.coeff_count() ||
            poly_to_evaluate_at.significant_coeff_bit_count() > coeff_modulus.significant_bit_count())
        {
            throw invalid_argument("poly_to_evaluate_at is not reduced");
        }

        int poly_to_eval_coeff_uint64_count = poly_to_evaluate.coeff_uint64_count();
        int coeff_modulus_bit_count = coeff_modulus.significant_bit_count();

        if (poly_to_evaluate.is_zero())
        {
            destination.set_zero();
        }

        if (poly_to_evaluate_at.is_zero())
        {
            destination.resize(1, coeff_modulus_bit_count);
            modulo_uint(poly_to_evaluate.data(), poly_to_eval_coeff_uint64_count, 
                Modulus(coeff_modulus.data(), coeff_modulus.uint64_count(), pool), 
                destination.data(), pool);
            return;
        }

        ConstPointer poly_to_eval_ptr = duplicate_poly_if_needed(poly_to_evaluate, poly_modulus.coeff_count(), coeff_modulus.uint64_count(), false, pool);
        ConstPointer poly_to_eval_at_ptr = duplicate_poly_if_needed(poly_to_evaluate_at, poly_modulus.coeff_count(), coeff_modulus.uint64_count(), false, pool);

        destination.resize(poly_modulus.coeff_count(), coeff_modulus_bit_count);

        util::poly_eval_poly_polymod_coeffmod(poly_to_eval_ptr.get(), poly_to_eval_at_ptr.get(),
            PolyModulus(poly_modulus.data(), poly_modulus.coeff_count(), poly_modulus.coeff_uint64_count()), 
            Modulus(coeff_modulus.data(), coeff_modulus.uint64_count(), pool),
            destination.data(), pool);
    }

    BigPoly poly_eval_poly_polymod_coeffmod(const BigPoly &poly_to_evaluate, const BigPoly &poly_to_evaluate_at, 
        const BigPoly &poly_modulus, const BigUInt &coeff_modulus, const MemoryPoolHandle &pool)
    {
        BigPoly result;
        poly_eval_poly_polymod_coeffmod(poly_to_evaluate, poly_to_evaluate_at, poly_modulus, coeff_modulus, result, pool);
        return result;
    }
    
    void poly_eval_uint_mod(const BigPoly &poly_to_evaluate, const BigUInt &value, const BigUInt &modulus, 
        BigUInt &destination, const MemoryPoolHandle &pool)
    {
        if (poly_to_evaluate.significant_coeff_bit_count() > modulus.significant_bit_count())
        {
            throw invalid_argument("poly_to_evaluate is not reduced");
        }
        if (value.significant_bit_count() > modulus.significant_bit_count())
        {
            throw invalid_argument("value is not reduced");
        }
        if (!pool)
        {
            throw invalid_argument("pool is uninitialized");
        }

        int poly_to_eval_coeff_uint64_count = poly_to_evaluate.coeff_uint64_count();
        int modulus_bit_count = modulus.significant_bit_count();

        if (poly_to_evaluate.is_zero())
        {
            destination.set_zero();
        }

        if (value.is_zero())
        {
            destination.resize(modulus_bit_count);
            modulo_uint(poly_to_evaluate.data(), poly_to_eval_coeff_uint64_count,
                Modulus(modulus.data(), modulus.uint64_count(), pool), 
                destination.data(), pool);
            return;
        }

        ConstPointer value_ptr = duplicate_uint_if_needed(value, modulus.uint64_count(), false, pool);

        destination.resize(modulus_bit_count);

        util::poly_eval_uint_mod(poly_to_evaluate.data(), poly_to_evaluate.coeff_count(), value_ptr.get(), 
            Modulus(modulus.data(), modulus.uint64_count(), pool), destination.data(), pool);
    }

    BigUInt poly_eval_uint_mod(const BigPoly &poly_to_evaluate, const BigUInt &value, const BigUInt &modulus, const MemoryPoolHandle &pool)
    {
        BigUInt result;
        poly_eval_uint_mod(poly_to_evaluate, value, modulus, result, pool);
        return result;
    }

    BigUInt exponentiate_uint_mod(const BigUInt &operand, const BigUInt &exponent, const BigUInt &modulus, const MemoryPoolHandle &pool)
    {
        BigUInt result(modulus.significant_bit_count());
        exponentiate_uint_mod(operand, exponent, modulus, result, pool);
        return result;
    }

    BigPoly exponentiate_poly_polymod_coeffmod(const BigPoly &operand, const BigUInt &exponent,
        const BigPoly &poly_modulus, const BigUInt &coeff_modulus, const MemoryPoolHandle &pool)
    {
        BigPoly result(poly_modulus.coeff_count(), coeff_modulus.significant_bit_count());
        exponentiate_poly_polymod_coeffmod(operand, exponent, poly_modulus, coeff_modulus, result, pool);
        return result;
    }

    //vector<uint64_t> compute_lift_uint64(long p, long n) 
    //{
    //    if (n < 1)
    //    {
    //        invalid_argument("n should be positive");
    //    }

    //    MemoryPoolHandle pool_;

    //    BigPoly X("1x^1");
    //    BigPoly result("1");
    //    BigUInt power_of_n_plus_one("1");
    //    for (int i = 0; i < n + 1; i++)	power_of_n_plus_one *= p;
    //    Modulus power_of_n_plus_one_(power_of_n_plus_one.data(), power_of_n_plus_one.uint64_count());

    //    // Set result = X^p
    //    for (int i = 0; i < p; i++)
    //    {
    //        result = bigpolyarith.multiply(result, X);
    //    }

    //    BigUInt power("1");
    //    BigPoly fi;
    //    BigPoly power_of_i_fi;
    //    for (int i = 1; i < n + 1; i++)
    //    {
    //        power *= p; // power = p^i
    //        fi = fpoly_ep(p, i, n); // get fpoly_ep
    //        modulo_poly_coeffs(fi.data(), fi.coeff_count(), power_of_n_plus_one_, pool_);
    //        power_of_i_fi.resize(fi.coeff_count(), fi.coeff_bit_count() + power.bit_count());
    //        multiply_poly_scalar_coeffmod(fi.data(), fi.coeff_count(), power.data(), power_of_n_plus_one_, power_of_i_fi.data(), pool_);
    //        sub_poly_poly_coeffmod(result.data(), power_of_i_fi.data(), power_of_i_fi.coeff_count(), power_of_n_plus_one.data(), power_of_i_fi.coeff_uint64_count(), result.data());
    //    }
    //    vector<uint64_t> lift;
    //    lift.resize(result.coeff_count());
    //    for (long i = 0; i < result.coeff_count(); i++) {
    //        lift[i] = *(result[i].data());
    //    }
    //    return lift;
    //}

    //vector<uint64_t> compute_remainlsd_uint64(long p, long n) {
    //    if (n < 2)
    //    {
    //        invalid_argument("n should be larger than 1");
    //    }

    //    long degree = (n - 1) * (p - 1) + 1;
    //    BigUInt power_of_n("1");
    //    for (int i = 0; i < n; i++)
    //    {
    //        power_of_n *= p;
    //    }
    //    Modulus mod_(power_of_n.data(), power_of_n.uint64_count());
    //    MemoryPoolHandle pool_;

    //    BigPoly X("1x^1");
    //    BigPoly one("1");
    //    BigPoly term("1x^1");
    //    BigPoly prod("1x^1");

    //    BigPoly result;
    //    BigUInt coeff;
    //    BigPoly amcoeff_times_prod;

    //    for (int i = 1; i < p; i++)
    //    {
    //        term = bigpolyarith.sub(term, one, power_of_n); // (X-i)
    //        prod = bigpolyarith.multiply(prod, term, power_of_n); // X(X-1)*...*(X-i)
    //    }

    //    coeff = amcoeff_divided_by_factorial(p, p, n);
    //    amcoeff_times_prod.resize(prod.coeff_count(), mod_.significant_bit_count());
    //    multiply_poly_scalar_coeffmod(prod.data(), prod.coeff_count(), coeff.data(), mod_, amcoeff_times_prod.data(), pool_);
    //    result.duplicate_from(amcoeff_times_prod);

    //    for (long m = p + 1; m <= (n - 1)*(p - 1) + 1; m++)
    //    {
    //        // Compute f(x) *= (X - m + 1)
    //        term = bigpolyarith.sub(term, one, power_of_n);
    //        prod = bigpolyarith.multiply(prod, term, power_of_n);
    //        // Compute g(x) = amcoeff(m,p) * f(x)
    //        coeff = amcoeff_divided_by_factorial(m, p, n);
    //        //amcoeff_times_prod.resize(prod.coeff_count(), prod.coeff_bit_count() + coeff.bit_count());
    //        amcoeff_times_prod.resize(prod.coeff_count(), mod_.significant_bit_count());
    //        multiply_poly_scalar_coeffmod(prod.data(), prod.coeff_count(), coeff.data(), mod_, amcoeff_times_prod.data(), pool_);
    //        // result += g(x)
    //        result = bigpolyarith.add(result, amcoeff_times_prod, power_of_n);
    //    }
    //    result = bigpolyarith.sub(X, result, power_of_n);
    //    vector<uint64_t> remainlsd;
    //    remainlsd.resize(result.coeff_count());
    //    for (long i = 0; i < result.coeff_count(); i++) {
    //        remainlsd[i] = *(result[i].data());
    //    }
    //    return remainlsd;
    //}

    //BigPoly fpoly_ep(long base, long exponent, long n)
    //{
    //    MemoryPoolHandle pool_;

    //    long p = base;
    //    long e = exponent;
    //    BigUInt power_of_e("1");
    //    for (int i = 0; i < e; i++)	power_of_e *= p;
    //    BigUInt power_of_n_plus_one("1");
    //    for (int i = 0; i < n + 1; i++) power_of_n_plus_one *= p;
    //    Modulus mod_(power_of_n_plus_one.data(), power_of_n_plus_one.uint64_count());

    //    vector<BigUInt> y;
    //    for (long i = 0; i < p; i++)
    //    {
    //        BigUInt v("1");
    //        if (i == 0) {
    //            v = 0;
    //        }
    //        else {
    //            for (long j = 0; j < p; j++) v *= i;
    //        }

    //        if (v < power_of_e)
    //        {
    //            BigUInt zero("0");
    //            y.push_back(zero);
    //        }
    //        else
    //        {
    //            BigUInt temp = v / power_of_e;
    //            temp %= p;
    //            y.push_back(temp);
    //        }
    //    }

    //    vector<BigPoly> terms(p); // X, X-1, ... , X-(p-1)
    //    BigPoly X("1x^1");
    //    BigPoly one("1");

    //    terms[0] = X;
    //    for (int i = 1; i < p; i++)
    //    {
    //        terms[i] = bigpolyarith.sub(terms[i - 1], one, power_of_n_plus_one);
    //    }

    //    BigPoly result(p + 1, power_of_n_plus_one.bit_count(), "0");
    //    BigUInt inv;
    //    for (int i = 0; i < p; i++)
    //    {
    //        BigPoly prod("1");
    //        BigUInt prod_biguint(power_of_n_plus_one.bit_count(), "1");
    //        BigUInt diff("0");
    //        for (int j = 0; j < p; j++)
    //        {
    //            if (j != i)
    //            {
    //                if (j > i) {
    //                    diff = power_of_n_plus_one - j;
    //                    diff += i;
    //                    prod_biguint *= diff; // prod_{j!=i} (i-j)
    //                    prod_biguint %= power_of_n_plus_one;
    //                }
    //                else {
    //                    diff = i - j;
    //                    prod_biguint *= diff; // prod_{j!=i} (i-j)
    //                    prod_biguint %= power_of_n_plus_one;
    //                }
    //                prod = bigpolyarith.multiply(prod, terms[j], power_of_n_plus_one); // prod_{j!=i}(X-j)
    //            }
    //        }
    //        BigUInt test;
    //        test = prod_biguint % power_of_n_plus_one;
    //        inv = test.modinv(power_of_n_plus_one);
    //        inv *= y[i];
    //        inv %= power_of_n_plus_one;
    //        BigPoly inv_times_prod;
    //        inv_times_prod.resize(prod.coeff_count(), power_of_n_plus_one.bit_count());
    //        multiply_poly_scalar_coeffmod(prod.data(), prod.coeff_count(), inv.data(), mod_, inv_times_prod.data(), pool_);
    //        bigpolyarith.add(result, inv_times_prod, power_of_n_plus_one, result);
    //    }
    //    return result;
    //}

    //class BigInt 
    //{
    //public:

    //    int sign;
    //    BigUInt absval;

    //    BigInt() 
    //    {
    //        this->sign = 1;
    //        this->absval = 0;
    //    }

    //    BigInt(int sign, BigUInt val) 
    //    {
    //        this->sign = sign;
    //        this->absval = val;
    //    }

    //    BigInt operator +(const BigInt operand2) 
    //    {
    //        BigUInt val;
    //        if (this->sign == operand2.sign)
    //        {
    //            val = this->absval + operand2.absval;
    //            return BigInt(this->sign, val);
    //        }
    //        else 
    //        {
    //            if (this->absval > operand2.absval) 
    //            {
    //                val = this->absval - operand2.absval;
    //                return BigInt(this->sign, val);
    //            }
    //            else 
    //            {
    //                val = operand2.absval - this->absval;
    //                return BigInt(operand2.sign, val);
    //            }
    //        }
    //    }

    //    BigInt operator *(const uint64_t x) 
    //    {
    //        BigUInt val;
    //        val = this->absval * x;
    //        return BigInt(this->sign, val);
    //    }

    //    BigInt operator /(const uint64_t x) 
    //    {
    //        BigUInt val;
    //        val = this->absval / x;
    //        return BigInt(this->sign, val);
    //    }
    //};

    //BigUInt binomial(long n, long k) 
    //{
    //    BigUInt res("1");
    //    long temp;
    //    for (long i = 1; i <= k; i++) 
    //    {
    //        temp = n - i + 1;
    //        res *= temp;
    //        res /= i;
    //    }
    //    return res;
    //}

    //BigUInt amcoeff_divided_by_factorial(long m, long p, long e)
    //{
    //    // mod_ = p^e
    //    BigUInt mod_("1");
    //    for (int i = 0; i < e; i++) mod_ *= p;

    //    // compute m!
    //    BigUInt factorial("1");
    //    for (long i = 1; i <= m; i++)
    //    {
    //        factorial *= i;
    //    }

    //    // remove all primes in factorial
    //    int count = 0;
    //    while (1) 
    //    {
    //        if (factorial % p == 0) 
    //        {
    //            factorial /= p;
    //            count++;
    //        }
    //        else break;
    //    }

    //    // Compute amcoeff
    //    BigUInt zero("0");
    //    BigInt temp(1, zero);
    //    BigInt binom;
    //    long k = m / p;
    //    for (long i = 1; i <= k; i++)
    //    {
    //        if ((m - i*p) % 2 == 0)
    //        {
    //            BigInt binom(1, binomial(m - 1, m - i*p));
    //            temp = temp + binom;
    //        }
    //        else
    //        {
    //            BigInt binom(-1, binomial(m - 1, m - i*p));
    //            temp = temp + binom;
    //        }
    //    }
    //    temp = temp * p;

    //    // remove same amount of primes = count
    //    for (long i = 0; i < count; i++) {
    //        temp = temp / p;
    //    }

    //    BigUInt GCD = gcd(factorial, temp.absval);
    //    factorial /= GCD;
    //    BigUInt abs_amcoeff;
    //    abs_amcoeff = temp.absval / GCD;

    //    if (temp.sign == -1) {
    //        factorial %= mod_;
    //        abs_amcoeff %= mod_;
    //        abs_amcoeff = mod_ - abs_amcoeff;
    //        abs_amcoeff *= factorial.modinv(mod_);
    //        abs_amcoeff %= mod_;
    //    }
    //    else {
    //        factorial %= mod_;
    //        abs_amcoeff %= mod_;
    //        abs_amcoeff *= factorial.modinv(mod_);
    //        abs_amcoeff %= mod_;
    //    }
    //    return abs_amcoeff;
    //}

    //BigUInt gcd(BigUInt &a, BigUInt &b)
    //{
    //    BigUInt c;
    //    if (a < b)
    //    {
    //        return gcd(b, a);
    //    }
    //    if (a == b)
    //    {
    //        return a;
    //    }
    //    if (b.is_zero())
    //    {
    //        return a;
    //    }
    //    c = a % b;

    //    return gcd(b, c);
    //}

    //int optimal_parameter_paterson(int degree)
    //{
    //    int minimum = (1 << 30);
    //    int min_k = 0;
    //    int min_m;
    //    int upper_bound = 2 * static_cast<int>(floor(sqrt(static_cast<double>(degree))));

    //    for (int k = 1; k < upper_bound; k++)
    //    {
    //        int m = 1;
    //        while (1)
    //        {
    //            int degree_prime = k * ((1 << m) - 1);
    //            if (degree_prime > degree)
    //            {
    //                int number_of_mult = (k - 1) + 2 * (m - 1) + ((1 << (m - 1)) - 1);
    //                if (number_of_mult < minimum)
    //                {
    //                    minimum = number_of_mult;
    //                    min_k = k;
    //                    min_m = m;
    //                }
    //                break;
    //            }
    //            else
    //            {
    //                m++;
    //            }
    //        }
    //    }
    //    return min_k;
    //}
}