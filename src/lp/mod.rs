#![allow(dead_code)]

use faer::{Col, sparse::SparseColMat};

use crate::{E, I};

/// A linear program in standard form:
///
/// ```text
/// min  c^T x
/// s.t. A x = b
///      l <= x <= u
/// ```
#[allow(non_snake_case)]
#[derive(Clone, Debug)]
pub struct LinearProgram {
    pub n_vars: usize,
    pub n_cons: usize,
    /// Objective function coefficients.
    pub c: Col<E>,
    /// Constraint matrix (sparse, column-major).
    pub A: SparseColMat<I, E>,
    /// Right-hand side of the equality constraints.
    pub b: Col<E>,
    /// Lower bounds on the variables.
    pub l: Col<E>,
    /// Upper bounds on the variables.
    pub u: Col<E>,
}

#[allow(non_snake_case)]
impl LinearProgram {
    /// Creates a new linear program from the objective, constraints, and bounds.
    pub fn new(c: Col<E>, A: SparseColMat<I, E>, b: Col<E>, l: Col<E>, u: Col<E>) -> Self {
        let n_vars = c.nrows();
        let n_cons = b.nrows();
        Self {
            n_vars,
            n_cons,
            c,
            A,
            b,
            l,
            u,
        }
    }

    /// Returns the number of variables (columns of `A`).
    pub fn get_n_vars(&self) -> usize {
        self.n_vars
    }

    /// Returns the number of constraints (rows of `A`).
    pub fn get_n_cons(&self) -> usize {
        self.n_cons
    }

    /// Returns `(n_vars, n_cons)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.get_n_vars(), self.get_n_cons())
    }

    pub fn objective(&self) -> &Col<E> {
        &self.c
    }

    pub fn constraints(&self) -> &SparseColMat<I, E> {
        &self.A
    }

    pub fn rhs(&self) -> &Col<E> {
        &self.b
    }

    pub fn lower_bounds(&self) -> &Col<E> {
        &self.l
    }

    pub fn upper_bounds(&self) -> &Col<E> {
        &self.u
    }

    pub fn get_objective_value(&self, x: &Col<E>) -> E {
        self.c.transpose() * x
    }

    pub fn get_constraint_values(&self, x: &Col<E>) -> Col<E> {
        self.A.as_ref() * x - &self.b
    }
}
