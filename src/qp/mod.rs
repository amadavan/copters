use faer::{Col, sparse::SparseColMat};
use problemo::Problem;
use problemo::common::IntoCommonProblem;

use crate::OptimizationProgram;
use crate::linalg::vector_ops::cwise_multiply_finite;
use crate::nlp::NonlinearProgram;
use crate::{
    E, I, Solver, SolverOptions,
    linalg::cholesky::{SimplicialSparseCholesky, SupernodalSparseCholesky},
};

pub mod mpc;

/// A linear program in standard form:
///
/// ```text
/// min  c^T x
/// s.t. A x = b
///      l <= x <= u
/// ```
#[allow(non_snake_case)]
pub struct QuadraticProgram {
    Q: SparseColMat<I, E>,
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
        Self { Q, c, A, b, l, u }
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

    pub fn get_linear_objective(&self) -> &Col<E> {
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

    pub fn solver_builder<'a>(&'a self) -> QPSolverBuilder<'a> {
        QPSolverBuilder::new().with_lp(self)
    }
}

impl OptimizationProgram for QuadraticProgram {
    fn compute_residual(&self, state: &crate::SolverState) -> crate::Residual {
        crate::Residual {
            dual_feasibility: -&self.Q * &state.x - &self.c
                + self.A.transpose() * &state.y
                + &state.z_l
                + &state.z_u,
            primal_feasibility: self.A.as_ref() * &state.x - &self.b,
            cs_lower: -cwise_multiply_finite(state.z_l.as_ref(), (&state.x - &self.l).as_ref()),
            cs_upper: -cwise_multiply_finite(state.z_u.as_ref(), (&state.x - &self.u).as_ref()),
        }
    }
}

#[allow(non_snake_case, unused)]
impl From<QuadraticProgram> for NonlinearProgram {
    fn from(qp: QuadraticProgram) -> Self {
        let n = qp.get_n_vars();
        let m = qp.get_n_cons();

        let Q = qp.Q.clone();
        let c = qp.c.clone();
        let A = qp.A.clone();
        let b = qp.b.clone();
        let l = qp.l.clone();
        let u = qp.u.clone();

        let Q2 = Q.clone();
        let c2 = c.clone();
        let A2 = A.clone();

        let f = Box::new(move |x: &Col<E>| 0.5 * x.transpose() * &Q * x + c.transpose() * x);
        let g = Box::new(move |x: &Col<E>| A.clone() * x - &b);
        let df = Box::new(move |x: &Col<E>| Q2.clone() * x + c2.clone());
        let dg = Box::new(move |_: &Col<E>| A2.clone());

        NonlinearProgram::new_boxed(n, m, f, g, df, dg, None, Some(l), Some(u))
    }
}

/// Trait for solvers that operate on a [`QuadraticProgram`].
pub trait QPSolver<'a>: Solver {
    /// Creates a new solver instance for the given linear program and options.
    fn new(lp: &'a QuadraticProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;
}

#[derive(Copy, Clone)]
pub enum QPSolverType {
    MpcSimplicialCholesky,
    MpcSupernodalCholesky,
    MpcSimplicialLu,
}

pub struct QPSolverBuilder<'a> {
    lp: Option<&'a QuadraticProgram>,
    solver_type: Option<QPSolverType>,
    options: SolverOptions,
}

impl<'a> QPSolverBuilder<'a> {
    pub fn new() -> Self {
        Self {
            lp: None,
            solver_type: None,
            options: SolverOptions::new(),
        }
    }

    pub fn with_lp(mut self, lp: &'a QuadraticProgram) -> Self {
        self.lp = Some(lp);
        self
    }

    pub fn with_solver(mut self, solver_type: QPSolverType) -> Self {
        self.solver_type = Some(solver_type);
        self
    }

    pub fn with_options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn build(self) -> Result<Box<dyn QPSolver<'a> + 'a>, Problem> {
        let lp = self
            .lp
            .ok_or_else(|| "Linear program must be provided".gloss())?;
        let solver_type = self
            .solver_type
            .ok_or_else(|| "Solver type must be specified".gloss())?;

        match solver_type {
            QPSolverType::MpcSimplicialCholesky => {
                Ok(Box::new(mpc::MehrotraPredictorCorrector::<
                    'a,
                    SimplicialSparseCholesky,
                    mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
                    mpc::mu_update::AdaptiveMuUpdate<'a>,
                >::new(lp, &self.options)))
            }
            QPSolverType::MpcSupernodalCholesky => {
                Ok(Box::new(mpc::MehrotraPredictorCorrector::<
                    'a,
                    SupernodalSparseCholesky,
                    mpc::augmented_system::StandardSystem<'a, SupernodalSparseCholesky>,
                    mpc::mu_update::AdaptiveMuUpdate<'a>,
                >::new(lp, &self.options)))
            }
            QPSolverType::MpcSimplicialLu => Ok(Box::new(mpc::MehrotraPredictorCorrector::<
                'a,
                SimplicialSparseCholesky,
                mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
                mpc::mu_update::AdaptiveMuUpdate<'a>,
            >::new(lp, &self.options))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::OnceLock;

    use faer::{
        Col, ColRef,
        sparse::{SparseColMat, Triplet},
    };
    use rstest::{fixture, rstest};
    use rstest_reuse::{apply, template};

    use crate::{
        E, SolverHooks, SolverOptions, SolverState,
        callback::ConvergenceOutput,
        terminators::{ConvergenceTerminator, Terminator},
    };

    #[template]
    #[rstest]
    pub fn solver_types(
        #[values(
            QPSolverType::MpcSimplicialCholesky,
            QPSolverType::MpcSupernodalCholesky,
            QPSolverType::MpcSimplicialLu
        )]
        solver_type: QPSolverType,
    ) {
    }

    #[fixture]
    #[allow(non_snake_case)]
    fn build_simple_qp() -> &'static QuadraticProgram {
        static QP: OnceLock<QuadraticProgram> = OnceLock::new();
        QP.get_or_init(|| {
            let Q = SparseColMat::try_new_from_triplets(
                3,
                3,
                &[
                    Triplet::new(0, 0, 2.0),
                    Triplet::new(1, 1, 2.0),
                    Triplet::new(2, 2, 2.0),
                ],
            )
            .unwrap();
            let c = ColRef::<E>::from_slice(&[0.0; 3]).to_owned();
            let A = SparseColMat::try_new_from_triplets(
                2,
                3,
                &[
                    Triplet::new(0, 0, 1.0),
                    Triplet::new(0, 1, 1.0),
                    Triplet::new(1, 1, 1.0),
                    Triplet::new(1, 2, 1.0),
                ],
            )
            .unwrap();
            let b = ColRef::<E>::from_slice(&[1.0; 2]).to_owned();
            let l = Col::<E>::zeros(3);
            let u = ColRef::<E>::from_slice(&[f64::INFINITY; 3]).to_owned();

            QuadraticProgram::new(Q, c, A, b, l, u)
        })
    }

    #[fixture]
    fn build_options() -> &'static SolverOptions {
        static OPTIONS: OnceLock<SolverOptions> = OnceLock::new();
        OPTIONS.get_or_init(|| {
            let mut options = SolverOptions::new();
            let _ = options.set_option("max_iterations", 1000);
            let _ = options.set_option("tolerance", 1e-8);
            options
        })
    }

    #[apply(solver_types)]
    fn test_solver_instances(
        #[values(build_simple_qp())] qp: &'static QuadraticProgram,
        solver_type: QPSolverType,
    ) {
        let mut state = SolverState::new(
            Col::ones(qp.get_n_vars()),
            Col::ones(qp.get_n_cons()),
            Col::ones(qp.get_n_vars()),
            -Col::<E>::ones(qp.get_n_vars()),
        );

        let options = SolverOptions::new();

        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut solver = QuadraticProgram::solver_builder(qp)
            .with_solver(solver_type)
            .with_options(options.clone())
            .build()
            .unwrap();
        let status = solver.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }
}
