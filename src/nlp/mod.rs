pub mod gd;
pub mod ipm;

use faer::{Col, sparse::SparseColMat};

use crate::{E, I};

/// A nonlinear program of the form:
///
/// ```text
///   min  f(x)
///   s.t. g(x) = 0
///        l <= x <= u
/// ```
///
/// where `f` is the objective function, `g` are equality constraints,
/// `df` is the gradient of `f`, `dg` is the Jacobian of `g`, and `h` is the
/// (optional) Hessian of the Lagrangian. The vectors `l` and `u` are optional
/// lower and upper bounds on the decision variables.
#[allow(unused)]
pub struct NonlinearProgram {
    /// Number of decision variables.
    n_var: I,
    /// Number of equality constraints.
    n_cons: I,

    /// Objective function `f(x) -> scalar`.
    f: fn(&Col<E>) -> E,
    /// Equality constraint function `g(x) -> Col`.
    g: fn(&Col<E>) -> Col<E>,
    /// Gradient of the objective `∇f(x)`.
    df: fn(&Col<E>) -> Col<E>,
    /// Jacobian of the constraints `∇g(x)` (sparse).
    dg: fn(&Col<E>) -> SparseColMat<I, E>,
    /// Hessian of the Lagrangian `∇²L(x, y)` (optional, sparse).
    h: Option<fn(&Col<E>, &Col<E>) -> SparseColMat<I, E>>,

    /// Lower bounds on the decision variables (optional).
    l: Option<Col<E>>,
    /// Upper bounds on the decision variables (optional).
    u: Option<Col<E>>,
}

#[allow(unused)]
impl NonlinearProgram {
    /// Creates a new nonlinear program from its component functions and bounds.
    pub fn new(
        n_var: I,
        n_cons: I,
        f: fn(&Col<E>) -> E,
        g: fn(&Col<E>) -> Col<E>,
        df: fn(&Col<E>) -> Col<E>,
        dg: fn(&Col<E>) -> SparseColMat<I, E>,
        h: Option<fn(&Col<E>, &Col<E>) -> SparseColMat<I, E>>,
        l: Option<Col<E>>,
        u: Option<Col<E>>,
    ) -> Self {
        Self {
            n_var,
            n_cons,
            f,
            g,
            df,
            dg,
            h,
            l,
            u,
        }
    }

    pub fn f(&self, x: &Col<E>) -> E {
        (self.f)(x)
    }

    pub fn g(&self, x: &Col<E>) -> Col<E> {
        (self.g)(x)
    }

    pub fn df(&self, x: &Col<E>) -> Col<E> {
        (self.df)(x)
    }

    pub fn dg(&self, x: &Col<E>) -> SparseColMat<I, E> {
        (self.dg)(x)
    }

    pub fn h(&self, x: &Col<E>, y: &Col<E>) -> Option<SparseColMat<I, E>> {
        if let Some(h_eval) = self.h {
            return Some((h_eval)(x, y));
        }
        None
    }

    pub fn l(&self) -> Option<&Col<E>> {
        self.l.as_ref()
    }

    pub fn u(&self) -> Option<&Col<E>> {
        self.u.as_ref()
    }
}
