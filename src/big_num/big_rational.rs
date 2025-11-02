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