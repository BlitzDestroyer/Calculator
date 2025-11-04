use crate::big_num::big_int::BigInt;

#[derive(Debug, Clone)]
pub struct BigRational {
    pub numerator: BigInt,
    pub denominator: BigInt,
}

impl BigRational {
    pub const fn new(numerator: BigInt, denominator: BigInt) -> Self {
        Self { numerator, denominator }
    }

    pub fn reciprocal(&self) -> BigRational {
        BigRational {
            numerator: self.denominator.clone(),
            denominator: self.numerator.clone(),
        }
    }

    pub fn normalize(self) -> BigRational {
        let num_sign = self.numerator.is_positive() != self.denominator.is_positive();
        let gcd = self.numerator.gcd(&self.denominator);
        let numerator = self.numerator / &gcd;
        let denominator = self.denominator / gcd;
        BigRational::new(numerator, if num_sign { -denominator } else { denominator })
    }

    pub fn pow(self, exp: BigInt) -> BigRational {
        if exp.is_negative() {
            let exp = exp.abs();
            BigRational::new(
                self.denominator.pow(exp.clone()).unwrap(),
                self.numerator.pow(exp).unwrap(),
            )
        }
        else {
            BigRational::new(
                self.numerator.pow(exp.clone()).unwrap(),
                self.denominator.pow(exp).unwrap(),
            )
        }
    }

    pub(in crate::big_num) fn add_numerator(self, other: BigInt) -> BigRational {
        BigRational::new(
            self.numerator + other,
            self.denominator,
        ).normalize()
    }

    pub(in crate::big_num) fn sub_numerator(self, other: BigInt) -> BigRational {
        BigRational::new(
            self.numerator - other,
            self.denominator,
        ).normalize()
    }

    pub(in crate::big_num) fn add_denominator(self, other: BigInt) -> BigRational {
        BigRational::new(
            self.numerator,
            self.denominator + other,
        ).normalize()
    }

    pub(in crate::big_num) fn sub_denominator(self, other: BigInt) -> BigRational {
        BigRational::new(
            self.numerator,
            self.denominator - other,
        ).normalize()
    }
}

impl std::fmt::Display for BigRational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Handle zero denominator just in case
        if self.denominator.is_zero() {
            return write!(f, "NaN");
        }

        // Integer and remainder
        let (int_part, mut remainder) = self.numerator.clone().try_div_mod(self.denominator.clone())
            .map_err(|_| std::fmt::Error)?;

        // Determine precision
        let precision = f.precision().unwrap_or(20); // default to 20 digits

        // Start with integer part
        let mut result = format!("{}", int_part);

        // If there is a remainder, compute fractional part
        if !remainder.is_zero() && precision > 0 {
            result.push('.');

            for _ in 0..precision {
                remainder = remainder * 10;
                let (digit, new_rem) = remainder.clone().try_div_mod(self.denominator.clone())
                    .map_err(|_| std::fmt::Error)?;

                result.push_str(&format!("{}", digit));
                remainder = new_rem;

                if remainder.is_zero() {
                    break;
                }
            }
        }

        write!(f, "{}", result)
    }
}

impl From<BigInt> for BigRational {
    fn from(value: BigInt) -> Self {
        Self {
            numerator: value,
            denominator: BigInt::one(),
        }
    }
}

impl std::ops::Add for BigRational {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        BigRational::new(
            self.numerator * &rhs.denominator + rhs.numerator * &self.denominator,
            self.denominator * rhs.denominator,
        ).normalize()
    }
}

impl std::ops::Sub for BigRational {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        BigRational::new(
            self.numerator * &rhs.denominator - rhs.numerator * &self.denominator,
            self.denominator * rhs.denominator,
        ).normalize()
    }
}

impl std::ops::Mul for BigRational {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        BigRational::new(
            self.numerator * rhs.numerator,
            self.denominator * rhs.denominator,
        ).normalize()
    }
}

impl std::ops::Div for BigRational {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        BigRational::new(
            self.numerator * rhs.denominator,
            self.denominator * rhs.numerator,
        ).normalize()
    }
}

impl std::ops::Rem for BigRational {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        BigRational::new(
            self.numerator % rhs.numerator,
            self.denominator % rhs.denominator,
        ).normalize()
    }
}

fn factorial(n: u64) -> BigInt {
    let mut result = BigInt::one();
    for i in 2..=n {
        result = result * BigInt::from_unsigned(i);
    }
    result
}

fn pi(iterations: u64) -> BigRational {
    let c = BigInt::from_unsigned(640320u64);
    let c3 = c.clone().pow(3.into()).unwrap();
    let mut sum = BigRational::from(BigInt::zero());

    for k in 0..iterations {
        // sign = (-1)^k
        let sign = if k % 2 == 0 { BigInt::one() } else { -BigInt::one() };

        // numerator = (6k)! * (13591409 + 545140134*k)
        let num = factorial(6 * k)
            * BigInt::from_signed(13591409i64 + 545140134i64 * k as i64)
            * sign;

        // denominator = (3k)! * (k!)^3 * 640320^(3k)
        println!("Calculating term for k={}", k);
        let den = factorial(3 * k)
            * factorial(k).pow(3.into()).unwrap()
            * c3.clone().pow(BigInt::from_unsigned(k)).unwrap();

        let term = BigRational::new(num, den);
        sum = sum + term;
    }

    // Constant multiplier: 12 / 640320^(3/2)
    // (640320^(3/2) = 640320^1.5 = 640320 * sqrt(640320))
    // sqrt(640320) is irrational, so we can approximate as rational with tolerance
    // but since you want rational arithmetic only, represent constant as BigRational(12, 640320^1)

    // So 1/pi = sum * 12 / (640320^(3/2))
    // We can rationally approximate sqrt(640320) using a fixed precision BigRational later.
    // For now, we’ll treat 640320^(3/2) ≈ 640320^3 for testing scale.

    let constant_num = BigInt::from_signed(12);
    let constant_den = c.clone().pow(BigInt::one()).unwrap(); // rough sqrt term omitted

    let pi_inv = sum * BigRational::new(constant_num, constant_den);
    pi_inv.reciprocal().normalize()
}

fn pi2(iteration: u64) -> BigRational {
    let c = BigInt::from_unsigned(640320u64);
    let mut sum = None;
    for k in 0..iteration {
        let sign = if k % 2 == 0 { BigInt::one() } else { -BigInt::one() };

        let num = factorial(6 * k)
            * BigInt::from_signed(13591409i64 + 545140134i64 * k as i64)
            * sign;

        let den = factorial(3 * k)
            * factorial(k).pow(3.into()).unwrap()
            * c.clone().pow(BigInt::from_unsigned(3 * k)).unwrap();

        let term = BigRational::new(num, den);
        sum = match sum {
            Some(s) => Some(s + term),
            None => Some(term),
        };
    }

    sum.unwrap_or_else(|| BigRational::new(BigInt::zero(), BigInt::one()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_big_rational_display() {
        let num = BigInt::from_signed(22);
        let den = BigInt::from_signed(7);
        let rational = BigRational::new(num, den);
        assert_eq!(format!("{:.10}", rational), "3.1428571428");
    }

    #[test]
    fn test_pi_approximation() {
        let pi_approx = pi2(10); // 10 iterations
        let pi_str = format!("{:.20}", pi_approx);
        let pi_actual = "3.14159265358979323846"; // 20 digits of actual Pi
        // Compare first 20 digits
        assert_eq!(&pi_str[..22], pi_actual);
    }
}