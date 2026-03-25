// pub mod ipm;

use faer::{Col, ColRef, sparse::SparseColMat};
use macros::use_option;

use crate::{E, I, IterativeSolver, OptimizationProgram, SolverOptions};

pub mod gd;

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
// #[use_option(name = "nlp_solver_type", type_ = crate::nlp::NLPSolverType, default = "gradient_descent", description = "Type of NLP solver to use.")]
pub struct NonlinearProgram {
    /// Number of decision variables.
    n_var: I,
    /// Number of equality constraints.
    n_cons: I,

    /// Objective function `f(x) -> scalar`.
    f: Box<dyn Fn(&ColRef<E>) -> E>,
    /// Equality constraint function `g(x) -> Col`.
    g: Box<dyn Fn(&ColRef<E>) -> Col<E>>,
    /// Gradient of the objective `∇f(x)`.
    df: Box<dyn Fn(&ColRef<E>) -> Col<E>>,
    /// Jacobian of the constraints `∇g(x)` (sparse).
    dg: Box<dyn Fn(&ColRef<E>) -> SparseColMat<I, E>>,
    /// Hessian of the Lagrangian `∇²L(x, y)` (optional, sparse).
    h: Option<Box<dyn Fn(&ColRef<E>, &ColRef<E>) -> SparseColMat<I, E>>>,

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
        f: impl Fn(&ColRef<E>) -> E + 'static,
        g: impl Fn(&ColRef<E>) -> Col<E> + 'static,
        df: impl Fn(&ColRef<E>) -> Col<E> + 'static,
        dg: impl Fn(&ColRef<E>) -> SparseColMat<I, E> + 'static,
        h: Option<impl Fn(&ColRef<E>, &ColRef<E>) -> SparseColMat<I, E> + 'static>,
        l: Option<Col<E>>,
        u: Option<Col<E>>,
    ) -> Self {
        Self {
            n_var,
            n_cons,
            f: Box::new(f),
            g: Box::new(g),
            df: Box::new(df),
            dg: Box::new(dg),
            h: h.map(|h_box| {
                Box::new(h_box) as Box<dyn Fn(&ColRef<E>, &ColRef<E>) -> SparseColMat<I, E>>
            }),
            l,
            u,
        }
    }

    pub fn f(&self, x: &ColRef<E>) -> E {
        (self.f)(x)
    }

    pub fn g(&self, x: &ColRef<E>) -> Col<E> {
        (self.g)(x)
    }

    pub fn df(&self, x: &ColRef<E>) -> Col<E> {
        (self.df)(x)
    }

    pub fn dg(&self, x: &ColRef<E>) -> SparseColMat<I, E> {
        (self.dg)(x)
    }

    pub fn h(&self, x: &ColRef<E>, y: &ColRef<E>) -> Option<SparseColMat<I, E>> {
        if let Some(h_eval) = &self.h {
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

impl OptimizationProgram for NonlinearProgram {}

pub trait NLPSolver<'a>: IterativeSolver {
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;
    // Define any additional methods specific to NLP solvers here.
}
