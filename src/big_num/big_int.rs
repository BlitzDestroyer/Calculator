use thiserror::Error;

use crate::{big_num::num_traits::{SignedInt, UnsignedInt, Int}, debug_println};

const DIGIT_MAX: u32 = 999_999_999;
const DIGIT_BASE: u32 = 1_000_000_000;
const DIGIT_U64_MAX: u64 = 999_999_999_999_999_99;
const DIGIT_U64_BASE: u64 = 1_000_000_000_000_000_00;

#[derive(Debug, Error)]
pub enum BigIntParseError {
    #[error("Invalid character in input string")]
    InvalidCharacter(char),
    #[error("Empty input string")]
    EmptyInput,
}

#[derive(Debug, Error)]
pub enum BigIntComputeError {
    #[error("Exponentiation with negative exponent is not supported")]
    NegativeExponent,
    #[error("Division by zero")]
    DivisionByZero,
}

#[derive(Debug, Clone, Copy)]
enum DigitsDirty {
    None,
    B10,
    B2,
}

#[derive(Debug, Clone)]
pub struct BigInt {
    sign: bool, // true for positive, false for negative
    digits_b10: Vec<u32>,
    digits_b2: Vec<u32>, // TODO: implement binary representation
    dirty: DigitsDirty,
}

impl BigInt {
    // Caution: digits_b10 must equal digits_b2 in their respective bases
    const fn new(sign: bool, digits_b10: Vec<u32>, digits_b2: Vec<u32>) -> Self {
        Self { sign, digits_b10, digits_b2, dirty: DigitsDirty::None }
    }

    pub const fn from_b10(sign: bool, digits: Vec<u32>) -> Self {
        Self { sign, digits_b10: digits, digits_b2: Vec::new(), dirty: DigitsDirty::B2 }
    }

    pub fn from_b2(sign: bool, digits: Vec<u32>) -> Self {
        Self { sign, digits_b10: Vec::new(), digits_b2: digits, dirty: DigitsDirty::B10 }
    }

    pub fn zero() -> Self {
        Self::new(true, vec![0], vec![0])
    }

    pub fn one() -> Self {
        Self::new(true, vec![1], vec![1])
    }

    pub fn from_unsigned<T: UnsignedInt>(value: T) -> Self
    {
        let value_u128 = value.to_u128().unwrap();
        if value_u128 > DIGIT_U64_MAX as u128 {
            let (hh, hl, lh, ll) = split_u128(value_u128);
            let mut digits = Vec::new();
            if hh > 0 { digits.push(hh); }
            if hl > 0 || !digits.is_empty() { digits.push(hl); }
            if lh > 0 || !digits.is_empty() { digits.push(lh); }
            digits.push(ll);
            Self::from_b10(true, digits)
        } 
        else if value_u128 > DIGIT_MAX as u128 {
            let (high, low) = split_u64(value_u128 as u64);
            let mut digits = Vec::new();
            if high > 0 { digits.push(high); }
            digits.push(low);
            Self::from_b10(true, digits)
        }
        else {
            Self::from_b10(true, vec![value_u128 as u32])
        }
    }

    pub fn from_signed<T: SignedInt>(value: T) -> Self 
    {
        if value.is_positive() {
            Self::from_unsigned(value.to_u128().unwrap())
        }
        else {
            let abs_value = value.wrapping_neg().to_u128().unwrap();
            let mut big_int = Self::from_unsigned(abs_value);
            big_int.sign = false;
            big_int
        }
    }

    pub fn parse(value: &str) -> Result<Self, BigIntParseError> {
        let mut chars = value.chars();
        let first_char = chars.next();
        let (sign, mut buf) = match first_char {
            Some('+') => (true, String::new()),
            Some('-') => (false, String::new()),
            Some(c) if c.is_digit(10) => (true, first_char.unwrap().to_string()),
            Some(c) => return Err(BigIntParseError::InvalidCharacter(c)),
            None => return Err(BigIntParseError::EmptyInput),
        };

        let mut digits = Vec::new();
        let mut exp_digits = Vec::new();
        let mut exp_buf = String::new();
        let mut in_exponent = false;
        for c in chars {
            match c {
                '_' => continue, // underscores are allowed as digit separators
                'e' | 'E' => {
                    if in_exponent {
                        return Err(BigIntParseError::InvalidCharacter(c));
                    }

                    in_exponent = true;
                },
                c if c.is_digit(10) => {
                    if in_exponent {
                        exp_buf.push(c);
                    }
                    else {
                        buf.push(c)
                    }
                },
                _ => return Err(BigIntParseError::InvalidCharacter(c)),
            }
        }


        let chunks = reverse_chunk_chars(&buf, 9);
        for chunk in chunks {
            let digit = chunk.parse::<u64>().unwrap();
            debug_println!("Parsed chunk: {}, digit: {}", chunk, digit);
            let (high, low) = split_u64(digit);
            if high > 0 {
                digits.push(high);
            }

            digits.push(low);
        }

        if !exp_buf.is_empty() {
            let chunks = reverse_chunk_chars(&exp_buf, 9);
            for chunk in chunks {
                let digit = chunk.parse::<u64>().unwrap();
                debug_println!("Parsed chunk: {}, exp digit: {}", chunk, digit);
                let (high, low) = split_u64(digit);
                if high > 0 {
                    exp_digits.push(high);
                }

                exp_digits.push(low);
            }
        }

        Ok(Self::from_b10(sign, digits))
    }

    pub fn is_zero(&self) -> bool {
        self.digits_b10.is_empty() || match self.dirty {
            DigitsDirty::None | DigitsDirty::B10 => self.digits_b2[0] == 0,
            DigitsDirty::B2 => self.digits_b10[0] == 0,
        }
    }

    pub fn is_positive(&self) -> bool {
        self.sign && !self.is_zero()
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        !self.is_positive()
    }

    pub fn is_even(&self) -> bool {
        if self.is_zero() {
            return true;
        }

        match self.dirty {
            DigitsDirty::None | DigitsDirty::B10 => {
                //debug_println!("is_even: {} & 1 = {}", self.digits_b2[0], self.digits_b2[0] & 1);
                self.digits_b2[0] & 1 == 0
            },
            DigitsDirty::B2 => {
                //debug_println!("is_even: {} & 1 = {}", self.digits_b10[0], self.digits_b10[0] & 1);
                self.digits_b10[0] & 1 == 0
            },
        }
    }

    #[inline]
    pub fn is_odd(&self) -> bool {
        !self.is_even()
    }

    fn clean_digits(&mut self) {
        match self.dirty {
            DigitsDirty::None => (),
            DigitsDirty::B10 => self.clean_digits_b10(),
            DigitsDirty::B2 => self.clean_digits_b2(),
        }
    }

    fn clean_digits_b10(&mut self) {
        if !matches!(self.dirty, DigitsDirty::B10) {
            return;
        }

        todo!()
    }

    fn clean_digits_b2(&mut self) {
        if !matches!(self.dirty, DigitsDirty::B2) {
            return;
        }

        todo!()
    }

    fn inner_add(&self, rhs: &Self) -> Self {
        if self.sign == rhs.sign {
            BigInt::from_b10(self.sign, add_least_sig_vec(&self.digits_b10, &rhs.digits_b10))
        }
        else {
            match cmp_least_sig_vec(&self.digits_b10, &rhs.digits_b10) {
                std::cmp::Ordering::Greater => {
                    let mut result = BigInt::from_b10(self.sign, sub_least_sig_vec(&self.digits_b10, &rhs.digits_b10));
                    result.normalize_zero();
                    result
                },
                std::cmp::Ordering::Less => {
                    let mut result = BigInt::from_b10(rhs.sign, sub_least_sig_vec(&rhs.digits_b10, &self.digits_b10));
                    result.normalize_zero();
                    result
                },
                std::cmp::Ordering::Equal => BigInt::zero(),
            }
        }
    }

    fn inner_mul(&self, rhs: &Self) -> Self {
        if self.is_zero() || rhs.is_zero() {
            return BigInt::zero();
        }

        let sign = self.sign == rhs.sign;
        let digits = karatsuba_mul(&self.digits_b10, &rhs.digits_b10);
        BigInt::from_b10(sign, digits)
    }

    pub fn abs(self) -> Self {
        match self.dirty {
            DigitsDirty::None | DigitsDirty::B10 => Self::from_b2(true, self.digits_b2),
            DigitsDirty::B2 => Self::from_b10(true, self.digits_b10),
        }
    }

    pub fn try_div_mod(mut self, divisor: BigInt) -> Result<(BigInt, BigInt), BigIntComputeError> {
        if divisor.is_zero() {
            return Err(BigIntComputeError::DivisionByZero);
        }

        self.clean_digits_b10();

        match self.cmp(&divisor) {
            std::cmp::Ordering::Less => return Ok((BigInt::zero(), self)),
            std::cmp::Ordering::Equal => return Ok((BigInt::one(), BigInt::zero())),
            std::cmp::Ordering::Greater => (),
        }

        let a = &self.digits_b10;
        let b = &divisor.digits_b10;

        let n = b.len();
        let m = a.len() - n;

        let shift = 0;
        let mut u = shl_digits(a, shift);
        let v = shl_digits(b, shift);

        u.push(0);

        let mut q = vec![0u32; m + 1];
        let base_u128: u128 = DIGIT_BASE as u128;
        let base_u64:  u64  = DIGIT_BASE as u64;

        for j in (0..=m).rev() {
            // ---- D3. trial q̂ and r̂ from the top 2 limbs ----
            let ujn  = u[j + n] as u128;
            let ujn1 = u[j + n - 1] as u128;
            let ujn2 = if n >= 2 { u[j + n - 2] as u128 } else { 0 };

            let v1 = v[n - 1] as u128;                 // MS limb of divisor
            let v2 = if n >= 2 { v[n - 2] as u128 } else { 0 };

            let mut qhat = ((ujn * base_u128) + ujn1) / v1;
            let mut rhat = ((ujn * base_u128) + ujn1) - (qhat * v1);

            if qhat >= base_u128 {
                qhat = base_u128 - 1;
                rhat += v1;
            }

            // ---- D4. at most two checks to correct q̂ ----
            while (qhat * v2) > (rhat * base_u128 + ujn2) {
                qhat -= 1;
                rhat += v1;
                if rhat >= base_u128 {
                    break;
                }
            }

            // ---- D4/D5. multiply subtract: u[j..j+n] -= q̂ * v ----
            // Use base-10^9 arithmetic with carry and underflow accounting.
            let mut carry: u64 = 0;
            for i in 0..n {
                // p = q̂ * v[i] + carry  (full precision)
                let p      = (qhat as u128) * (v[i] as u128) + (carry as u128);
                let p_low  = (p % base_u128) as u32;
                let p_high = (p / base_u128) as u64;

                // subtract p_low from u[j+i], tracking underflow into carry
                if u[j + i] >= p_low {
                    u[j + i] = u[j + i] - p_low;
                    carry = p_high;
                }
                else {
                    // borrow one base from the next limb
                    let t = (u[j + i] as u64 + base_u64) - (p_low as u64);
                    u[j + i] = t as u32;
                    carry = p_high + 1;
                }
            }

            // subtract final carry from u[j+n]
            if (u[j + n] as u64) >= carry {
                u[j + n] = (u[j + n] as u64 - carry) as u32;
            }
            else {
                // ---- D5. subtraction underflowed → q̂ was one too big.
                // Decrement q̂ and add divisor back into u[j..j+n].
                qhat -= 1;

                let mut c: u64 = 0;
                for i in 0..n {
                    let s = (u[j + i] as u64) + (v[i] as u64) + c;
                    if s >= base_u64 {
                        u[j + i] = (s - base_u64) as u32;
                        c = 1;
                    }
                    else {
                        u[j + i] = s as u32;
                        c = 0;
                    }
                }
                // add leftover carry into top limb
                u[j + n] = (u[j + n] as u64 + c) as u32;
            }

            q[j] = qhat as u32;

            debug_println!("After sub: u[{}..]: {:?}", j, &u[j..j+n+1]);
        }

        let mut r = shr_digits(&u[..n], shift);
        while r.len() > 1 && *r.last().unwrap() == 0 {
            r.pop();
        }

        while q.len() > 1 && *q.last().unwrap() == 0 {
            q.pop();
        }

        let q_sign = self.sign == divisor.sign;
        let r_sign = self.sign;

        Ok((BigInt::from_b10(q_sign, q), BigInt::from_b10(r_sign, r)))
    }

    pub fn pow(self, exp: BigInt) -> Result<BigInt, BigIntComputeError> {
        if exp.is_zero() {
            return Ok(BigInt::one());
        }

        if exp.is_negative() {
            return Err(BigIntComputeError::NegativeExponent);
        }

        let mut result = BigInt::one();
        let mut exp = exp;

        let mut base = self;
        while !exp.is_zero() {
            if exp.is_odd() {
                result = result * base.clone();
            }
            
            base = base.clone() * base;
            exp = exp / 2;
            debug_println!("Intermediate pow result: {}, exp: {}, mult: {}", result, exp, base);
        }

        Ok(result)
    }

    #[inline]
    pub fn gcd(&self, b: &BigInt) -> BigInt {
        BigInt::gcd_lehmer(self.clone(), b.clone())
    }

    pub fn gcd_euclid(&self, b: &BigInt) -> BigInt {
        let mut a = self.clone();
        let mut b = b.clone();
        while !b.is_zero() {
            let r = a.clone().try_div_mod(b.clone()).unwrap().1;
            a = b;
            b = r;
        }
        a
    }

    pub fn gcd_lehmer(mut a: BigInt, mut b: BigInt) -> BigInt {
        // Step 1.  Ensure a ≥ b ≥ 0
        if a < b {
            std::mem::swap(&mut a, &mut b);
        }
        if b.is_zero() {
            return a;
        }

        // Step 2.  Main loop
        loop {
            // If b fits in one limb → do a normal small-int GCD
            if b.digits_b10.len() == 1 {
                return gcd_small(a, b);
            }

            // ---- Lehmer phase ----
            // Extract top one or two limbs of a,b
            let (mut ah, mut bh) = leading_digits(&a, &b);

            // Use small 128-bit integers for the “matrix” iteration
            let mut m00: i128 = 1;
            let mut m01: i128 = 0;
            let mut m10: i128 = 0;
            let mut m11: i128 = 1;

            // repeatedly perform integer steps while the 64-bit approximation holds
            while bh as i128 + m10 != 0
                && bh as i128 + m11 != 0
            {
                let q = (ah as i128 + m00) / (bh as i128 + m10);
                let q2 = (ah as i128 + m01) / (bh as i128 + m11);
                if q != q2 {
                    break;
                }

                // update transformation matrix (simulate Euclid)
                let t = m00 - q * m01;
                m00 = m01;
                m01 = t;
                let t = m10 - q * m11;
                m10 = m11;
                m11 = t;

                let tmp = ah as i128 - q * bh as i128;
                ah = bh;
                bh = tmp as u128;
            }

            // ---- Apply matrix to full BigInts ----
            if m10 != 0 {
                let a_new = m00 * &a - m01 * &b;
                let b_new = m10 * a - m11 * b;
                a = a_new.abs();
                b = b_new.abs();
            }
            else {
                // The 64-bit model broke down → fall back to full divmod once
                let r = a.clone().try_div_mod(b.clone()).unwrap().1;
                a = b;
                b = r;
            }

            if b.is_zero() {
                break;
            }
            if a < b {
                std::mem::swap(&mut a, &mut b);
            }
        }

        a
    }

    // fn mul_small(&self, k: i128) -> BigInt {
    //     let mut carry: i128 = 0;
    //     let base = DIGIT_BASE as i128;
    //     let mut out = Vec::with_capacity(self.digits_b10.len() + 1);

    //     for &d in &self.digits_b10 {
    //         let val = d as i128 * k + carry;
    //         carry = val.div_euclid(base);
    //         out.push((val.rem_euclid(base)) as u32);
    //     }
    //     if carry != 0 {
    //         out.push(carry as u32);
    //     }
    //     BigInt::from_b10(self.sign, out)
    // }

    fn normalize_zero(&mut self) {
        match self.dirty {
            DigitsDirty::None => {
                if self.digits_b10.iter().all(|&d| d == 0) {
                    self.sign = true;
                    self.digits_b10 = vec![0];
                    self.digits_b2 = vec![0];
                }
            },
            DigitsDirty::B10 => {
                if self.digits_b2.iter().all(|&d| d == 0) {
                    self.sign = true;
                    self.digits_b2 = vec![0];
                    self.digits_b10 = vec![0];
                    self.dirty = DigitsDirty::None;
                }
            },
            DigitsDirty::B2 => {
                if self.digits_b10.iter().all(|&d| d == 0) {
                    self.sign = true;
                    self.digits_b10 = vec![0];
                    self.digits_b2 = vec![0];
                    self.dirty = DigitsDirty::None;
                }
            },
        }
    }
}

impl std::fmt::Display for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        if !self.sign {
            write!(f, "-")?;
        }

        if f.alternate() {
            let mut len = self.digits_b10.len();
            let mut digits = self.digits_b10.iter().peekable();
            let mut power = 0;
            while let Some(0) = digits.peek() {
                len -= 1;
                power += 9;
                digits.next();
            }

            let mut start = true;
            for (i, digit) in digits.rev().enumerate() {
                if start {
                    let formatted = format!("{}", digit);
                    let (first, rest) = formatted.split_at(1);
                    if rest.is_empty() {
                        write!(f, "{}", first)?;
                    }
                    else {
                        power += rest.len() as i32;
                        write!(f, "{}.{}", first, rest)?;
                    }
                    start = false;
                }
                else{
                    power += 9;
                    if i == len - 1 {
                        let formatted = format!("{}", digit);
                        let trimmed = formatted.trim_end_matches('0');
                        let trimmed_len = trimmed.len();
                        power -= 9 - trimmed_len as i32;
                        write!(f, "{}", trimmed)?;
                    }
                    else{
                        write!(f, "{}", digit)?;
                    }
                }
            }

            if power > 0 {
                write!(f, "e{}", power)?;
            }
        }
        else{
            for digit in self.digits_b10.iter().rev() {
                write!(f, "{}", digit)?;
            }
        }

        Ok(())
    }
}

impl From<&str> for BigInt {
    fn from(value: &str) -> Self {
        BigInt::parse(value).unwrap()
    }
}

impl<T: Int> From<T> for BigInt {
    fn from(value: T) -> Self {
        if T::min_value() < T::zero() {
            BigInt::from_signed(value.to_i128().unwrap())
        }
        else {
            BigInt::from_unsigned(value.to_u128().unwrap())
        }
    }
}

impl Eq for BigInt {}

impl PartialEq for BigInt {
    fn eq(&self, other: &Self) -> bool {
        self.sign == other.sign && self.digits_b10 == other.digits_b10
    }
}

impl Ord for BigInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.sign, other.sign) {
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (true, true) => cmp_least_sig_vec(&self.digits_b10, &other.digits_b10),
            (false, false) => cmp_least_sig_vec(&other.digits_b10, &self.digits_b10),
        }
    }
}

impl PartialOrd for BigInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::ops::Neg for BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            return self;
        }

        BigInt::from_b10(!self.sign, self.digits_b10)
    }
}

impl std::ops::Neg for &BigInt {
    type Output = BigInt;

    fn neg(self) -> Self::Output {
        if self.is_zero() {
            return self.clone();
        }

        BigInt::from_b10(!self.sign, self.digits_b10.clone())
    }
}

impl std::ops::Add for BigInt {
    type Output = BigInt;

    fn add(self, rhs: Self) -> Self::Output {
        self.inner_add(&rhs)
    }
}

impl std::ops::Add<&BigInt> for BigInt {
    type Output = BigInt;

    fn add(self, rhs: &BigInt) -> Self::Output {
        self.inner_add(rhs)
    }
}

impl<T: Int> std::ops::Add<T> for BigInt
{
    type Output = BigInt;

    fn add(self, rhs: T) -> Self::Output {
        if T::min_value() < T::zero() {
            let rhs = BigInt::from_signed(rhs.to_i128().unwrap());
            self + rhs
        }
        else {
            let rhs = BigInt::from_unsigned(rhs.to_u128().unwrap());
            self + rhs
        }
    }
}

macro_rules! impl_add_bigint_for_int {
    ($($t:ty),*) => {
        $(
            impl std::ops::Add<BigInt> for $t {
                type Output = BigInt;
                fn add(self, rhs: BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs + rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs + rhs
                    }
                }
            }

            impl std::ops::Add<&BigInt> for $t {
                type Output = BigInt;
                fn add(self, rhs: &BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs + rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs + rhs
                    }
                }
            }
        )*
    };
}

impl_add_bigint_for_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl std::ops::Sub for BigInt {
    type Output = BigInt;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Sub<&BigInt> for BigInt {
    type Output = BigInt;

    fn sub(self, rhs: &BigInt) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: Int> std::ops::Sub<T> for BigInt {
    type Output = BigInt;

    fn sub(self, rhs: T) -> Self::Output {
        if T::min_value() < T::zero() {
            let rhs = BigInt::from_signed(rhs.to_i128().unwrap());
            self - rhs
        }
        else {
            let rhs = BigInt::from_unsigned(rhs.to_u128().unwrap());
            self - rhs
        }
    }
}

macro_rules! impl_sub_bigint_for_int {
    ($($t:ty),*) => {
        $(
            impl std::ops::Sub<BigInt> for $t {
                type Output = BigInt;
                fn sub(self, rhs: BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs - rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs - rhs
                    }
                }
            }

            impl std::ops::Sub<&BigInt> for $t {
                type Output = BigInt;
                fn sub(self, rhs: &BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs - rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs - rhs
                    }
                }
            }
        )*
    };
}

impl_sub_bigint_for_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl std::ops::Mul for BigInt {
    type Output = BigInt;

    fn mul(self, rhs: Self) -> Self::Output {
        self.inner_mul(&rhs)
    }
}

impl std::ops::Mul<&BigInt> for BigInt {
    type Output = BigInt;

    fn mul(self, rhs: &BigInt) -> Self::Output {
        self.inner_mul(rhs)
    }
}

impl<T: Int> std::ops::Mul<T> for BigInt {
    type Output = BigInt;

    fn mul(self, rhs: T) -> Self::Output {
        if T::min_value() < T::zero() {
            let rhs = BigInt::from_signed(rhs.to_i128().unwrap());
            self * rhs
        }
        else {
            let rhs = BigInt::from_unsigned(rhs.to_u128().unwrap());
            self * rhs
        }
    }
}

macro_rules! impl_mul_bigint_for_int {
    ($($t:ty),*) => {
        $(
            impl std::ops::Mul<BigInt> for $t {
                type Output = BigInt;
                fn mul(self, rhs: BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs * rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs * rhs
                    }
                }
            }

            impl std::ops::Mul<&BigInt> for $t {
                type Output = BigInt;
                fn mul(self, rhs: &BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs * rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs * rhs
                    }
                }
            }
        )*
    };
}

impl_mul_bigint_for_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl std::ops::Div for BigInt {
    type Output = BigInt;

    fn div(self, rhs: Self) -> Self::Output {
        let (quotient, _remainder) = self.try_div_mod(rhs).unwrap();
        quotient
    }
}

impl std::ops::Div<&BigInt> for BigInt {
    type Output = BigInt;

    fn div(self, rhs: &BigInt) -> Self::Output {
        let (quotient, _remainder) = self.try_div_mod(rhs.clone()).unwrap();
        quotient
    }
}

impl<T: Int> std::ops::Div<T> for BigInt {
    type Output = BigInt;

    fn div(self, rhs: T) -> Self::Output {
        if T::min_value() < T::zero() {
            let rhs = BigInt::from_signed(rhs.to_i128().unwrap());
            self / rhs
        }
        else {
            let rhs = BigInt::from_unsigned(rhs.to_u128().unwrap());
            self / rhs
        }
    }
}

macro_rules! impl_div_bigint_for_int {
    ($($t:ty),*) => {
        $(
            impl std::ops::Div<BigInt> for $t {
                type Output = BigInt;
                fn div(self, rhs: BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs / rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs / rhs
                    }
                }
            }

            impl std::ops::Div<&BigInt> for $t {
                type Output = BigInt;
                fn div(self, rhs: &BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs / rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs / rhs
                    }
                }
            }
        )*
    };
}

impl_div_bigint_for_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl std::ops::Rem for BigInt {
    type Output = BigInt;

    fn rem(self, rhs: Self) -> Self::Output {
        let (_quotient, remainder) = self.try_div_mod(rhs).unwrap();
        remainder
    }
}

impl std::ops::Rem<&BigInt> for BigInt {
    type Output = BigInt;

    fn rem(self, rhs: &BigInt) -> Self::Output {
        let (_quotient, remainder) = self.try_div_mod(rhs.clone()).unwrap();
        remainder
    }
}

impl<T: Int> std::ops::Rem<T> for BigInt {
    type Output = BigInt;

    fn rem(self, rhs: T) -> Self::Output {
        if T::min_value() < T::zero() {
            let rhs = BigInt::from_signed(rhs.to_i128().unwrap());
            self % rhs
        }
        else {
            let rhs = BigInt::from_unsigned(rhs.to_u128().unwrap());
            self % rhs
        }
    }
}

macro_rules! impl_rem_bigint_for_int {
    ($($t:ty),*) => {
        $(
            impl std::ops::Rem<BigInt> for $t {
                type Output = BigInt;
                fn rem(self, rhs: BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs % rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs % rhs
                    }
                }
            }

            impl std::ops::Rem<&BigInt> for $t {
                type Output = BigInt;
                fn rem(self, rhs: &BigInt) -> Self::Output {
                    // Trait is impl for both signed and unsigned types, and the comp is necessary for signed types
                    #[allow(unused_comparisons)]
                    if <$t>::min_value() < 0 {
                        let lhs = BigInt::from_signed(self as i128);
                        lhs % rhs
                    }
                    else {
                        let lhs = BigInt::from_unsigned(self as u128);
                        lhs % rhs
                    }
                }
            }
        )*
    };
}

impl_rem_bigint_for_int!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

fn cmp_least_sig_vec(a: &Vec<u32>, b: &Vec<u32>) -> std::cmp::Ordering {
    let mut len_a = a.len();
    let mut len_b = b.len();

    let mut a = a.iter().rev().peekable();
    let mut b = b.iter().rev().peekable();
    while matches!(a.peek(), Some(&0)) {
        a.next();
        len_a -= 1;
    }

    while matches!(b.peek(), Some(&0)) {
        b.next();
        len_b -= 1;
    }
    
    if len_a != len_b {
        return len_a.cmp(&len_b);
    }

    for (digit_a, digit_b) in a.zip(b) {
        if digit_a != digit_b {
            return digit_a.cmp(digit_b);
        }
    }

    std::cmp::Ordering::Equal
}

fn add_least_sig_vec(a: &[u32], b: &[u32]) -> Vec<u32> {
    let max_len = std::cmp::max(a.len(), b.len());
    let mut result = Vec::with_capacity(max_len + 1);
    let mut carry = 0u64;
    for i in 0..max_len {
        let digit_a = a.get(i).unwrap_or(&0);
        let digit_b = b.get(i).unwrap_or(&0);

        let sum = *digit_a as u64 + *digit_b as u64 + carry;
        let (high, low) = split_u64(sum);
        result.push(low);
        carry = high as u64;
    }

    if carry > 0 {
        result.push(carry as u32);
    }

    result
}

fn sub_least_sig_vec(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = Vec::with_capacity(a.len());
    let mut borrow = 0i64;
    for i in 0..a.len() {
        let digit_a = a.get(i).unwrap_or(&0);
        let digit_b = b.get(i).unwrap_or(&0);

        let sub = (*digit_a as i64) - (*digit_b as i64) - borrow;
        if sub < 0 {
            result.push((sub + DIGIT_BASE as i64) as u32);
            borrow = 1;
        }
        else {
            result.push(sub as u32);
            borrow = 0;
        }
    }

    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }

    result
}

fn mul_least_sig_vec(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut result = vec![0u32; a.len() + b.len()];

    for (i, &digit_a) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &digit_b) in b.iter().enumerate() {
            let idx = i + j;
            let prod = digit_a as u64 * digit_b as u64 + result[idx] as u64 + carry;
            let (high, low) = split_u64(prod);
            result[idx] = low;
            carry = high as u64;
        }

        if carry > 0 {
            result[i + b.len()] += carry as u32;
        }
    }

    while result.len() > 1 && *result.last().unwrap() == 0 {
        result.pop();
    }

    result
}

fn karatsuba_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.len() == 1 || b.len() == 1 {
        return mul_least_sig_vec(a, b);
    }

    let mid = std::cmp::max(a.len(), b.len()) / 2;
    let (low_a, high_a) = a.split_at(mid);
    let (low_b, high_b) = b.split_at(mid);
    
    let z0 = karatsuba_mul(low_a, low_b);
    let z1 = karatsuba_mul(&add_least_sig_vec(low_a, high_a), &add_least_sig_vec(low_b, high_b));
    let z2 = karatsuba_mul(high_a, high_b);

    let p1 = shift_left(&z2, 2 * mid);
    let p2 = shift_left(&sub_least_sig_vec(&sub_least_sig_vec(&z1, &z2), &z0), mid);
    let result = add_least_sig_vec(&add_least_sig_vec(&p1, &p2), &z0);
    
    result
}

// TODO: Toom-Cook
// TODO: Schönhage–Strassen
// TODO: Fürer

fn shift_left(digits: &[u32], positions: usize) -> Vec<u32> {
    let mut result = vec![0u32; positions];
    result.extend_from_slice(digits);
    result
}

fn split_u64(value: u64) -> (u32, u32) {
    if value > DIGIT_MAX as u64 {
        let low = (value % DIGIT_BASE as u64) as u32;
        let high = (value / DIGIT_BASE as u64) as u32;
        (high, low)
    }
    else {
        (0, value as u32)
    }
}

fn split_u128(value: u128) -> (u32, u32, u32, u32) {
    if value > DIGIT_U64_MAX as u128 {
        let low = (value % DIGIT_U64_BASE as u128) as u64;
        let high = (value / DIGIT_U64_BASE as u128) as u64;
        let (high_high, high_low) = split_u64(high);
        let (low_high, low_low) = split_u64(low);
        (high_high, high_low, low_high, low_low)
    }
    else {
        let (low_high, low_low) = split_u64(value as u64);
        (0, 0, low_high, low_low)
    }
}

fn reverse_chunk_chars(s: &str, size: usize) -> Vec<String> {
    let chars: Vec<_> = s.chars().collect();
    let mut chunks = Vec::new();

    let mut i = chars.len();
    while i > 0 {
        let start = if i >= size { i - size } else { 0 };
        chunks.push(chars[start..i].iter().collect::<String>());
        i = start;
    }

    chunks
}

fn shl_digits(digits: &[u32], shift: u32) -> Vec<u32> {
    if shift == 0 { return digits.to_vec(); }
    let mut result = Vec::with_capacity(digits.len());
    let mut carry = 0u64;
    for &d in digits {
        let val = ((d as u64) << shift) + carry;
        result.push((val % DIGIT_BASE as u64) as u32);
        carry = val / DIGIT_BASE as u64;
    }

    if carry > 0 { 
        result.push(carry as u32); 
    }

    result
}

fn shr_digits(digits: &[u32], shift: u32) -> Vec<u32> {
    if shift == 0 { return digits.to_vec(); }
    let mut result = Vec::with_capacity(digits.len());
    let mut carry = 0u64;
    for &d in digits.iter().rev() {
        let val = (carry << 32) + d as u64;
        result.push((val >> shift) as u32);
        carry = val & ((1u64 << shift) - 1);
    }

    result.reverse();
    result
}

fn leading_digits(a: &BigInt, b: &BigInt) -> (u128, u128) {
    let base = DIGIT_BASE as u128;
    let n = b.digits_b10.len();

    let get = |x: &Vec<u32>, idx: usize| -> u128 {
        if idx >= x.len() { 0 } else { x[idx] as u128 }
    };

    // top two limbs of each operand
    let ah = get(&a.digits_b10, n) * base + get(&a.digits_b10, n - 1);
    let bh = get(&b.digits_b10, n - 1) * base + get(&b.digits_b10, n - 2);
    (ah, bh)
}

fn gcd_small(a: BigInt, b: BigInt) -> BigInt {
    let mut x = a.digits_b10[0] as u64;
    let mut y = b.digits_b10[0] as u64;

    while y != 0 {
        let temp = x % y;
        x = y;
        y = temp;
    }

    BigInt::from_b10(true, vec![x as u32])
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_big_int_parse() {
        let num = BigInt::parse("-12345678901234567890").unwrap();
        assert_eq!(num.sign, false);
        assert_eq!(num.digits_b10, vec![234567890, 345678901, 12]);
    }

    #[test]
    fn test_big_int_display() {
        let num = BigInt::parse("12345678901234567890").unwrap();
        assert_eq!(format!("{}", num), "12345678901234567890");
    }

    #[test]
    fn test_big_int_display_exponential() {
        let num = BigInt::parse("12_345678901_000000000").unwrap();
        assert_eq!(format!("{:#}", num), "1.2345678901e19");
    }

    #[test]
    fn test_big_int_addition() {
        let num1 = BigInt::parse("12345678901234567890").unwrap();
        let num2 = BigInt::parse("98765432109876543210").unwrap();
        let sum = num1 + num2;
        assert_eq!(format!("{}", sum), "111111111011111111100");
    }

    #[test]
    fn test_big_int_subtraction() {
        let num1 = BigInt::parse("98765432109876543210").unwrap();
        let num2 = BigInt::parse("12345678901234567890").unwrap();
        let diff = num1 - num2;
        assert_eq!(format!("{}", diff), "86419753208641975320");
    }

    // Test to ensure that subtraction resulting in zero normalizes correctly (i.e., sign is positive and digits is [0])
    #[test]
    fn test_big_int_subtraction_normalize() {
        let num1 = BigInt::parse("987654321098765432109876543210987654321098765432109876543210").unwrap();
        let num2 = BigInt::parse("987654321098765432109876543210987654321098765432109876543210").unwrap();
        let diff = num1 - num2;
        assert_eq!(format!("{}", diff), "0");
        assert_eq!(diff.sign, true);
        assert_eq!(diff.digits_b10, vec![0]);
    }

    #[test]
    fn test_big_int_multiplication() {
        let num1 = BigInt::parse("12345678901234567890").unwrap();
        let num2 = BigInt::parse("98765432109876543210").unwrap();
        let product = num1 * num2;
        assert_eq!(format!("{}", product), "1219326311370217952237463801111263526900");
    }

    #[test]
    fn test_big_int_division() {
        let num1 = BigInt::parse("1219326311370217952237463801111263526900").unwrap();
        let num2 = BigInt::parse("12345678901234567890").unwrap();
        let (quotient, remainder) = num1.try_div_mod(num2).unwrap();
        assert_eq!(format!("{}", quotient), "98765432109876543210");
        assert_eq!(format!("{}", remainder), "0");
    }

    #[test]
    fn test_big_int_remainder() {
        let num1 = BigInt::parse("1219326311370217952237463801111263526900").unwrap();
        let num2 = BigInt::parse("12345678901234567890").unwrap();
        let remainder = num1 % num2;
        assert_eq!(format!("{}", remainder), "0");
    }

    #[test]
    fn test_big_int_power() {
        let base = BigInt::parse("2").unwrap();
        let exponent = BigInt::parse("10").unwrap();
        let result = base.pow(exponent).unwrap();
        assert_eq!(format!("{}", result), "1024");
    }

    #[test]
    fn test_big_int_gcd() {
        let num1 = BigInt::parse("48").unwrap();
        let num2 = BigInt::parse("18").unwrap();
        let gcd = num1.gcd(&num2);
        assert_eq!(format!("{}", gcd), "6");
    }

    // TODO: Implement bitwise operators
}