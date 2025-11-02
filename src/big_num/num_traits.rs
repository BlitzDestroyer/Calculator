use num_traits::{PrimInt, Signed, ToPrimitive, Unsigned, WrappingNeg};

mod sealed {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for i128 {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for u128 {}
}

pub trait Int: sealed::Sealed + PrimInt + ToPrimitive {}

impl Int for i8 {}
impl Int for i16 {}
impl Int for i32 {}
impl Int for i64 {}
impl Int for i128 {}
impl Int for u8 {}
impl Int for u16 {}
impl Int for u32 {}
impl Int for u64 {}
impl Int for u128 {}

pub trait SignedInt: sealed::Sealed + PrimInt + Signed + ToPrimitive + WrappingNeg {}

impl SignedInt for i8 {}
impl SignedInt for i16 {}
impl SignedInt for i32 {}
impl SignedInt for i64 {}
impl SignedInt for i128 {}

pub trait UnsignedInt: sealed::Sealed + PrimInt + Unsigned + ToPrimitive {}

impl UnsignedInt for u8 {}
impl UnsignedInt for u16 {}
impl UnsignedInt for u32 {}
impl UnsignedInt for u64 {}
impl UnsignedInt for u128 {}