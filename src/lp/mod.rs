use faer::{Col, sparse::SparseColMat};

use crate::{E, I, IterativeSolver, SolverOptions};

pub mod mpc;

#[allow(non_snake_case)]
pub struct LinearProgram {
    // Fields representing the linear program
    c: Col<E>,
    A: SparseColMat<I, E>,
    b: Col<E>,
    // Additional fields for bounds, variable types, etc.
    l: Col<E>,
    u: Col<E>,
}

#[allow(non_snake_case)]
impl LinearProgram {
    pub fn new(c: Col<E>, A: SparseColMat<I, E>, b: Col<E>, l: Col<E>, u: Col<E>) -> Self {
        Self { c, A, b, l, u }
    }

    pub fn get_n_vars(&self) -> usize {
        self.c.nrows()
    }

    pub fn get_n_cons(&self) -> usize {
        self.b.nrows()
    }

    pub fn get_dims(&self) -> (usize, usize) {
        (self.get_n_vars(), self.get_n_cons())
    }
}

pub trait LinearProgramSolver<'a>: IterativeSolver {
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
        mpc::line_search::LPLineSearch<'a>,
    >;
    type MPCSupernodal<'a> = mpc::MehrotraPredictorCorrector<
        'a,
        SimplicialSparseCholesky,
        mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
        mpc::mu_update::AdaptiveMuUpdate<'a>,
        mpc::line_search::LPLineSearch<'a>,
    >;

    #[matrix_parameterized_test(
        types = (MPCSimplicial, MPCSupernodal),
        args = [build_simple_lp(),]
    )]
    fn test_lp<'a, T: LinearProgramSolver<'a>>(lp: &'a LinearProgram) {
        let mut state = SolverState::new(
            Col::ones(lp.c.nrows()),
            Col::ones(lp.b.nrows()),
            Col::ones(lp.c.nrows()),
            -Col::<E>::ones(lp.c.nrows()),
        );

        let mut options = SolverOptions::new();

        let mut properties = Properties {
            callback: Box::new(ConvergenceOutput::new(&options)),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut s = T::new(lp, &options);
        let status = s.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }
}
