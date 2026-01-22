use std::ops::Div;

use faer::Index;
use faer::traits::ComplexField;
use faer::traits::num_traits::{Float, PrimInt};

#[macro_use]
extern crate macros;

pub trait ElementType: ComplexField + Float + Div<Output = Self> + PrimInt {}
impl<T> ElementType for T where T: ComplexField + Float + Div<Output = T> + PrimInt {}

pub trait IndexType: Copy + PartialEq + Eq + Ord + Index {}
impl<T> IndexType for T where T: Copy + PartialEq + Eq + Ord + Index {}

pub type E = f64;
pub type I = usize;

pub mod linalg;
