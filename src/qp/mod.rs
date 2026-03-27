use faer::{Col, sparse::SparseColMat};

use crate::{E, I, lp::LinearProgram};

#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct QuadraticProgram {
    pub n_vars: usize,
    pub n_cons: usize,
    /// Quadratic objective matrix (symmetric, positive semidefinite).
    pub Q: SparseColMat<I, E>,
    /// Linear objective coefficients.
    pub c: Col<E>,
    /// Constraint matrix.
    pub A: SparseColMat<I, E>,
    /// Right-hand side of constraints.
    pub b: Col<E>,
    /// Lower bounds on variables.
    pub l: Col<E>,
    /// Upper bounds on variables.
    pub u: Col<E>,
}

#[allow(non_snake_case)]
impl QuadraticProgram {
    /// Creates a new quadratic program from the objective, constraints, and bounds.
    pub fn new(
        Q: SparseColMat<I, E>,
        c: Col<E>,
        A: SparseColMat<I, E>,
        b: Col<E>,
        l: Col<E>,
        u: Col<E>,
    ) -> Self {
        let n_vars = c.nrows();
        let n_cons = b.nrows();
        Self {
            n_vars,
            n_cons,
            Q,
            c,
            A,
            b,
            l,
            u,
        }
    }

    pub fn objective(&self) -> (&SparseColMat<I, E>, &Col<E>) {
        (&self.Q, &self.c)
    }

    pub fn constraints(&self) -> (&SparseColMat<I, E>, &Col<E>) {
        (&self.A, &self.b)
    }

    pub fn bounds(&self) -> (&Col<E>, &Col<E>) {
        (&self.l, &self.u)
    }

    pub fn dims(&self) -> (usize, usize) {
        (self.n_vars, self.n_cons)
    }

    pub fn get_n_vars(&self) -> usize {
        self.n_vars
    }

    pub fn get_n_cons(&self) -> usize {
        self.n_cons
    }
}

impl From<LinearProgram> for QuadraticProgram {
    fn from(lp: LinearProgram) -> Self {
        Self {
            n_vars: lp.n_vars,
            n_cons: lp.n_cons,
            Q: SparseColMat::try_new_from_nonnegative_triplets(lp.n_vars, lp.n_vars, &[]).unwrap(),
            c: lp.c,
            A: lp.A,
            b: lp.b,
            l: lp.l,
            u: lp.u,
        }
    }
}
