use faer::{Col, sparse::SparseColMat};
use problemo::Problem;
use problemo::common::IntoCommonProblem;

use crate::OptimizationProgram;
use crate::linalg::vector_ops::cwise_multiply_finite;
use crate::nlp::NonlinearProgram;
use crate::qp::QuadraticProgram;
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

    pub fn solver_builder<'a>(&'a self) -> LPSolverBuilder<'a> {
        LPSolverBuilder::new().with_lp(self)
    }

    pub fn get_objective_value(&self, x: &Col<E>) -> E {
        self.c.transpose() * x
    }

    pub fn get_constraint_values(&self, x: &Col<E>) -> Col<E> {
        self.A.as_ref() * x - &self.b
    }
}

#[allow(unused, non_snake_case)]
impl From<LinearProgram> for QuadraticProgram {
    fn from(lp: LinearProgram) -> Self {
        let n = lp.get_n_vars();
        let Q = SparseColMat::try_new_from_triplets(n, n, &[]).unwrap();
        QuadraticProgram::new(Q, lp.c, lp.A, lp.b, lp.l, lp.u)
    }
}

#[allow(unused, non_snake_case)]
impl From<&LinearProgram> for QuadraticProgram {
    fn from(lp: &LinearProgram) -> Self {
        let n = lp.get_n_vars();
        let Q = SparseColMat::try_new_from_triplets(n, n, &[]).unwrap();
        QuadraticProgram::new(
            Q,
            lp.c.clone(),
            lp.A.clone(),
            lp.b.clone(),
            lp.l.clone(),
            lp.u.clone(),
        )
    }
}

#[allow(unused, non_snake_case)]
impl From<LinearProgram> for NonlinearProgram {
    fn from(lp: LinearProgram) -> Self {
        let n = lp.get_n_vars();
        let m = lp.get_n_cons();

        let c = lp.get_objective().clone();
        let A = lp.get_constraint_matrix().clone();
        let b = lp.get_rhs().clone();
        let A2 = A.clone();
        let c2 = c.clone();

        let f = Box::new(move |x: &Col<E>| c.transpose() * x);
        let g = Box::new(move |x: &Col<E>| A.clone() * x - &b);
        let df = Box::new(move |x: &Col<E>| c2.clone());
        let dg = Box::new(move |_: &Col<E>| A2.clone());

        NonlinearProgram::new_boxed(n, m, f, g, df, dg, None, Some(lp.l), Some(lp.u))
    }
}

#[allow(unused, non_snake_case)]
impl From<&LinearProgram> for NonlinearProgram {
    fn from(lp: &LinearProgram) -> Self {
        let n = lp.get_n_vars();
        let m = lp.get_n_cons();

        let c = lp.get_objective().clone();
        let A = lp.get_constraint_matrix().clone();
        let b = lp.get_rhs().clone();
        let A2 = A.clone();
        let c2 = c.clone();

        let f = Box::new(move |x: &Col<E>| c.transpose() * x);
        let g = Box::new(move |x: &Col<E>| A.clone() * x - &b);
        let df = Box::new(move |x: &Col<E>| c2.clone());
        let dg = Box::new(move |_: &Col<E>| A2.clone());

        NonlinearProgram::new_boxed(
            n,
            m,
            f,
            g,
            df,
            dg,
            None,
            Some(lp.l.clone()),
            Some(lp.u.clone()),
        )
    }
}

impl OptimizationProgram for LinearProgram {
    fn compute_residual(&self, state: &crate::SolverState) -> crate::Residual {
        crate::Residual {
            dual_feasibility: -&self.c + self.A.transpose() * &state.y + &state.z_l + &state.z_u,
            primal_feasibility: self.A.as_ref() * &state.x - &self.b,
            cs_lower: -cwise_multiply_finite(state.z_l.as_ref(), (&state.x - &self.l).as_ref()),
            cs_upper: -cwise_multiply_finite(state.z_u.as_ref(), (&state.x - &self.u).as_ref()),
        }
    }
}

/// Trait for solvers that operate on a [`LinearProgram`].
pub trait LPSolver<'a>: Solver {
    /// Creates a new solver instance for the given linear program and options.
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;
}

#[derive(Copy, Clone)]
pub enum LPSolverType {
    MpcSimplicialCholesky,
    MpcSupernodalCholesky,
    MpcSimplicialLu,
}

pub struct LPSolverBuilder<'a> {
    lp: Option<&'a LinearProgram>,
    solver_type: Option<LPSolverType>,
    options: SolverOptions,
}

impl<'a> LPSolverBuilder<'a> {
    pub fn new() -> Self {
        Self {
            lp: None,
            solver_type: None,
            options: SolverOptions::new(),
        }
    }

    pub fn with_lp(mut self, lp: &'a LinearProgram) -> Self {
        self.lp = Some(lp);
        self
    }

    pub fn with_solver(mut self, solver_type: LPSolverType) -> Self {
        self.solver_type = Some(solver_type);
        self
    }

    pub fn with_options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn build(self) -> Result<Box<dyn LPSolver<'a> + 'a>, Problem> {
        let lp = self
            .lp
            .ok_or_else(|| "Linear program must be provided".gloss())?;
        let solver_type = self
            .solver_type
            .ok_or_else(|| "Solver type must be specified".gloss())?;

        match solver_type {
            LPSolverType::MpcSimplicialCholesky => {
                Ok(Box::new(mpc::MehrotraPredictorCorrector::<
                    'a,
                    SimplicialSparseCholesky,
                    mpc::augmented_system::SlackReducedSystem<'a, SimplicialSparseCholesky>,
                    mpc::mu_update::AdaptiveMuUpdate<'a>,
                >::new(lp.into(), &self.options)))
            }
            LPSolverType::MpcSupernodalCholesky => {
                Ok(Box::new(mpc::MehrotraPredictorCorrector::<
                    'a,
                    SupernodalSparseCholesky,
                    mpc::augmented_system::SlackReducedSystem<'a, SupernodalSparseCholesky>,
                    mpc::mu_update::AdaptiveMuUpdate<'a>,
                >::new(lp.into(), &self.options)))
            }
            LPSolverType::MpcSimplicialLu => {
                Ok(Box::new(mpc::MehrotraPredictorCorrector::<
                    'a,
                    SimplicialSparseCholesky,
                    mpc::augmented_system::SlackReducedSystem<'a, SimplicialSparseCholesky>,
                    mpc::mu_update::AdaptiveMuUpdate<'a>,
                >::new(lp.into(), &self.options)))
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::sync::OnceLock;

    use faer::{
        Col,
        sparse::{SparseColMat, Triplet},
    };
    use rstest::{fixture, rstest};
    use rstest_reuse::{apply, template};

    use crate::{
        E, I, SolverHooks, SolverOptions, SolverState, callback::ConvergenceOutput,
        lp::LinearProgram, terminators::ConvergenceTerminator,
    };

    #[template]
    #[rstest]
    pub fn solver_types(
        #[values(
            LPSolverType::MpcSimplicialCholesky,
            LPSolverType::MpcSupernodalCholesky,
            LPSolverType::MpcSimplicialLu
        )]
        solver_type: LPSolverType,
    ) {
    }

    #[fixture]
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
        #[values(build_simple_lp())] lp: &'static LinearProgram,
        solver_type: LPSolverType,
    ) {
        let mut state = SolverState::new(
            Col::ones(lp.c.nrows()),
            Col::ones(lp.b.nrows()),
            Col::ones(lp.c.nrows()),
            -Col::<E>::ones(lp.c.nrows()),
        );

        let options = SolverOptions::new();

        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut solver = LinearProgram::solver_builder(lp)
            .with_solver(solver_type)
            .with_options(options.clone())
            .build()
            .unwrap();
        let status = solver.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }
}
