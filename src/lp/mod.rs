use faer::{Col, sparse::SparseColMat};

use crate::{E, I, Solver, SolverOptions};

pub mod mpc;

/// A linear program in standard form:
///
/// ```text
/// min  c^T x
/// s.t. A x = b
///      l <= x <= u
/// ```
#[allow(non_snake_case)]
pub struct LinearProgram {
    /// Objective function coefficients.
    c: Col<E>,
    /// Constraint matrix (sparse, column-major).
    A: SparseColMat<I, E>,
    /// Right-hand side of the equality constraints.
    b: Col<E>,
    /// Lower bounds on the variables.
    l: Col<E>,
    /// Upper bounds on the variables.
    u: Col<E>,
}

#[allow(non_snake_case)]
impl LinearProgram {
    /// Creates a new linear program from the objective, constraints, and bounds.
    pub fn new(c: Col<E>, A: SparseColMat<I, E>, b: Col<E>, l: Col<E>, u: Col<E>) -> Self {
        Self { c, A, b, l, u }
    }

    /// Returns the number of variables (columns of `A`).
    pub fn get_n_vars(&self) -> usize {
        self.c.nrows()
    }

    /// Returns the number of constraints (rows of `A`).
    pub fn get_n_cons(&self) -> usize {
        self.b.nrows()
    }

    /// Returns `(n_vars, n_cons)`.
    pub fn get_dims(&self) -> (usize, usize) {
        (self.get_n_vars(), self.get_n_cons())
    }

    pub fn get_objective(&self) -> &Col<E> {
        &self.c
    }

    pub fn get_constraint_matrix(&self) -> &SparseColMat<I, E> {
        &self.A
    }

    pub fn get_rhs(&self) -> &Col<E> {
        &self.b
    }

    pub fn get_lower_bounds(&self) -> &Col<E> {
        &self.l
    }

    pub fn get_upper_bounds(&self) -> &Col<E> {
        &self.u
    }
}

/// Trait for solvers that operate on a [`LinearProgram`].
pub trait LinearProgramSolver<'a>: Solver {
    /// Creates a new solver instance for the given linear program and options.
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self;
}

#[cfg(test)]
mod test {
    use std::sync::OnceLock;

    use faer::{
        Col,
        sparse::{SparseColMat, Triplet},
    };
    use macros::matrix_parameterized_test;

    use crate::{
        E, I, Properties, SolverOptions, SolverState,
        callback::{Callback, ConvergenceOutput},
        linalg::cholesky::SimplicialSparseCholesky,
        lp::{LinearProgram, LinearProgramSolver, mpc},
        terminators::{ConvergenceTerminator, Terminator},
    };

    fn build_simple_lp() -> &'static LinearProgram {
        static LP: OnceLock<LinearProgram> = OnceLock::new();
        LP.get_or_init(|| {
            let a_triplets: [Triplet<I, I, E>; 9] = [
                Triplet::new(0, 0, -1.),
                Triplet::new(1, 0, 1.),
                Triplet::new(2, 0, -1.),
                Triplet::new(0, 1, -1.),
                Triplet::new(1, 1, -2.),
                Triplet::new(2, 1, 1.),
                Triplet::new(2, 2, 1.),
                Triplet::new(0, 3, 1.),
                Triplet::new(1, 4, 1.),
            ];
            let a = SparseColMat::try_new_from_triplets(3, 5, a_triplets.as_slice()).unwrap();

            LinearProgram::new(
                Col::from_fn(5, |i| [2., 1., 0., 0., 0.][i]),
                a,
                Col::from_fn(3, |i| [-2., 4., 1.][i]),
                Col::from_fn(5, |i| [-E::INFINITY, 0., 0., 0., 0.][i]),
                Col::from_fn(5, |i| {
                    [
                        E::INFINITY,
                        E::INFINITY,
                        E::INFINITY,
                        E::INFINITY,
                        E::INFINITY,
                    ][i]
                }),
            )
        })
    }

    type MPCSimplicial<'a> = mpc::MehrotraPredictorCorrector<
        'a,
        SimplicialSparseCholesky,
        mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
        mpc::mu_update::AdaptiveMuUpdate<'a>,
    >;
    type MPCSupernodal<'a> = mpc::MehrotraPredictorCorrector<
        'a,
        SimplicialSparseCholesky,
        mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
        mpc::mu_update::AdaptiveMuUpdate<'a>,
    >;

    #[matrix_parameterized_test(
        types = (MPCSimplicial, MPCSupernodal),
        named_args = [
            ("Simple LP", build_simple_lp()),
          ],
    )]
    fn test_lp<'a, T: LinearProgramSolver<'a>>(lp: &'a LinearProgram) {
        let mut state = SolverState::new(
            Col::ones(lp.c.nrows()),
            Col::ones(lp.b.nrows()),
            Col::ones(lp.c.nrows()),
            -Col::<E>::ones(lp.c.nrows()),
        );

        let options = SolverOptions::new();

        let mut properties = Properties {
            callback: Box::new(ConvergenceOutput::new(&options)),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut s = T::new(lp, &options);
        let status = s.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }
}
