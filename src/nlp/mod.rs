use faer::{Col, sparse::SparseColMat};

use crate::{E, I};

#[allow(dead_code)]
struct NonlinearProgram {
    n_var: I,
    n_cons: I,

    f: fn(Col<E>) -> E,
    g: fn(Col<E>) -> Col<E>,
    df: fn(Col<E>) -> Col<E>,
    dg: fn(Col<E>) -> SparseColMat<I, E>,
    h: Option<fn(Col<E>, Col<E>) -> SparseColMat<I, E>>,
}

#[allow(dead_code)]
impl NonlinearProgram {
    fn new(
        n_var: I,
        n_cons: I,
        f: fn(Col<E>) -> E,
        g: fn(Col<E>) -> Col<E>,
        df: fn(Col<E>) -> Col<E>,
        dg: fn(Col<E>) -> SparseColMat<I, E>,
        h: Option<fn(Col<E>, Col<E>) -> SparseColMat<I, E>>,
    ) -> Self {
        Self {
            n_var,
            n_cons,
            f,
            g,
            df,
            dg,
            h,
        }
    }

    fn f(self, x: Col<E>) -> E {
        (self.f)(x)
    }

    fn g(self, x: Col<E>) -> Col<E> {
        (self.g)(x)
    }

    fn df(self, x: Col<E>) -> Col<E> {
        (self.df)(x)
    }

    fn dg(self, x: Col<E>) -> SparseColMat<I, E> {
        (self.dg)(x)
    }

    fn h(self, x: Col<E>, y: Col<E>) -> Option<SparseColMat<I, E>> {
        if let Some(h_eval) = self.h {
            return Some((h_eval)(x, y));
        }
        None
    }
}
