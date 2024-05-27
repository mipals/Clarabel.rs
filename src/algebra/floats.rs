#![allow(non_snake_case)]
use num_traits::{Float, FloatConst, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display, LowerExp};

#[cfg(feature = "sdp")]
use crate::algebra::dense::BlasFloatT;

// PJG: this should be conditional
//#[cfg(feature = "serde")]
use serde;

/// Core traits for internal floating point values.
///
/// This trait defines a subset of bounds for `FloatT`, which is preferred
/// throughout for use in the solver.  When the "sdp" feature is enabled,
/// `FloatT` is additionally restricted to f32/f64 types supported by BLAS.
pub trait CoreFloatT:
    'static
    + Send
    + Float
    + FloatConst
    + NumAssign
    + Default
    + FromPrimitive
    + Display
    + LowerExp
    + Debug
    + Sized
    + serde::Serialize //PJG: make conditional
    + for<'a> serde::Deserialize<'a> //PJG: make conditional
{
}

impl<T> CoreFloatT for T where
    T: 'static
        + Send
        + Float
        + FloatConst
        + NumAssign
        + Default
        + FromPrimitive
        + Display
        + LowerExp
        + Debug
        + Sized
        + serde::Serialize //PJG: make conditional
        + for<'a> serde::Deserialize<'a> //PJG: make conditional
{
}

// if "sdp" is enabled, we must add an additional trait
// trait bound to restrict compilation for f32/f64 types
// since there is no BLAS support otherwise

// Define the documentation string as a macro so that FloatT gets documentation
// regardless of whether the "sdp" feature is enabled.
macro_rules! floatT_doc_header {
    () => {
        r#"Main trait for floating point types used in the Clarabel solver."#
    };
}
macro_rules! floatT_doc_long {
    () => {
        r#"All floating point calculations in Clarabel are represented internally on values 
         implementing the `FloatT` trait, with implementations provided only for f32 and f64 
         native types when compiled with BLAS/LAPACK support for SDPs. If SDP support is not 
         enabled then it should be possible to compile Clarabel to support any any other 
         floating point type provided that it satisfies the trait bounds of `CoreFloatT`. 
        \
        \
         `FloatT` relies on [`num_traits`](num_traits) for most of its constituent trait bounds."#
    };
}
cfg_if::cfg_if! {
    if #[cfg(not(feature="sdp"))] {
    #[doc = floatT_doc_header!()]
    ///
    #[doc = floatT_doc_long!()]
        pub trait FloatT: CoreFloatT {}
    } else{
        #[doc = floatT_doc_header!()]
        ///
        #[doc = floatT_doc_long!()]
        ///
        /// The trait bound `BlasFloatT` is only enforced when compiling with the `sdp` feature.
        pub trait FloatT: CoreFloatT + BlasFloatT {}
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature="sdp")] {
        impl<T> FloatT for T where T: CoreFloatT + BlasFloatT {}
    } else{
        impl<T> FloatT for T where T: CoreFloatT {}
    }
}

/// Trait for convering Rust primitives to [`FloatT`](crate::algebra::FloatT)
///
/// This convenience trait is implemented on f32/64 and u32/64.  This trait
/// is required internally by the solver for converting constant primitives
/// to [`FloatT`](crate::algebra::FloatT).  It is also used by the
/// [user settings](crate::solver::implementations::default::DefaultSettings)
/// for converting defaults of primitive type to [`FloatT`](crate::algebra::FloatT).

// NB: `AsFloatT` is a convenience trait for f32/64 and u32/64
// so that we can do things like (2.0).as_T() everywhere on
// constants, rather than the awful T::from_f32(2.0).unwrap()
pub trait AsFloatT<T>: 'static {
    fn as_T(&self) -> T;
}

macro_rules! impl_as_FloatT {
    ($ty:ty, $ident:ident) => {
        impl<T> AsFloatT<T> for $ty
        where
            T: std::ops::Mul<T, Output = T> + FromPrimitive + 'static,
        {
            #[inline]
            fn as_T(&self) -> T {
                T::$ident(*self).unwrap()
            }
        }
    };
}
impl_as_FloatT!(u32, from_u32);
impl_as_FloatT!(u64, from_u64);
impl_as_FloatT!(usize, from_usize);
impl_as_FloatT!(f32, from_f32);
impl_as_FloatT!(f64, from_f64);
