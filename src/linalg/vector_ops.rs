use std::f64::INFINITY;
use std::ops::{Div, Mul};

use faer::{Col, ColRef, unzip, zip};

use crate::E;

pub(crate) fn cwise_multiply<'a>(x1: ColRef<'a, E>, x2: ColRef<'a, E>) -> Col<E>
where
    E: Mul<Output = E>,
{
    let mut out = Col::<E>::zeros(x1.nrows());

    zip!(x1, x2, out.as_mut()).for_each(|unzip!(x1, x2, out)| *out = *x1 * *x2);

    out
}

pub(crate) fn cwise_multiply_positive<'a>(x1: ColRef<'a, E>, x2: ColRef<'a, E>) -> Col<E>
where
    E: Mul<Output = E> + PartialOrd,
{
    let mut out = Col::<E>::zeros(x1.nrows());

    zip!(x1, x2, out.as_mut()).for_each(|unzip!(x1, x2, out)| {
        let product = *x1 * *x2;
        *out = if product < E::from(0.) {
            E::from(0.)
        } else {
            product
        }
    });

    out
}

pub(crate) fn cwise_quotient<'a>(x1: ColRef<'a, E>, x2: ColRef<'a, E>) -> Col<E>
where
    E: Div<Output = E>,
{
    let mut out = Col::<E>::zeros(x1.nrows());

    zip!(x1, x2, out.as_mut()).for_each(|unzip!(x1, x2, out)| *out = *x1 / *x2);

    out
}

pub(crate) fn cwise_inverse<'a>(x: ColRef<'a, E>) -> Col<E>
where
    E: Div<Output = E>,
{
    let mut out = Col::<E>::zeros(x.nrows());

    zip!(x, out.as_mut()).for_each(|unzip!(x, out)| *out = E::from(1.) / *x);

    out
}

pub(crate) fn col_min<'a>(x: ColRef<'a, E>) -> E {
    let mut minimum = E::from(INFINITY);

    zip!(x).for_each(|unzip!(x)| minimum = E::min(minimum, *x));

    minimum
}

pub(crate) fn is_col_positive<'a>(x: ColRef<'a, E>) -> bool {
    let mut res = true;
    zip!(x).for_each(|unzip!(x)| {
        if *x <= E::from(0.) {
            res = false
        }
    });
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cwise_quotient() {
        let x1_data = [1.0, 2.0, 3.0];
        let x2_data = [4.0, 5.0, 6.0];
        let x1 = Col::from_fn(x1_data.len(), |i| x1_data[i]);
        let x2 = Col::from_fn(x2_data.len(), |i| x2_data[i]);
        let result = cwise_quotient(x1.as_ref(), x2.as_ref());
        let expected = [0.25, 0.4, 0.5];
        let expected_col = Col::from_fn(expected.len(), |i| expected[i]);
        assert_eq!(result, expected_col);
    }

    #[test]
    fn test_col_min() {
        let x1_data = [1.0, 2.0, 3.0];
        let x2_data = [1.0, -2.0, 3.0];
        let x1 = Col::from_fn(x1_data.len(), |i| x1_data[i]);
        let x2 = Col::from_fn(x2_data.len(), |i| x2_data[i]);
        assert!(is_col_positive(x1.as_ref()));
        assert!(!is_col_positive(x2.as_ref()));
    }

    #[test]
    fn test_is_col_positive() {
        let x1_data = [1.0, 2.0, 3.0];
        let x2_data = [1.0, -2.0, 3.0];
        let x1 = Col::from_fn(x1_data.len(), |i| x1_data[i]);
        let x2 = Col::from_fn(x2_data.len(), |i| x2_data[i]);
        assert!(is_col_positive(x1.as_ref()));
        assert!(!is_col_positive(x2.as_ref()));
    }
}
