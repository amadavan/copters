use faer::{Col, sparse::SparseColMat};
use problemo::{Problem, common::IntoCommonProblem};

use crate::{
    E, I, OptimizationProgram, Solver, SolverOptions,
    linalg::{
        cholesky::{SimplicialSparseCholesky, SupernodalSparseCholesky},
        lu::SimplicialSparseLu,
    },
    lp::LinearProgram,
    qp::pc::PredictorCorrector,
};

pub mod pc;

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

impl OptimizationProgram for QuadraticProgram {}

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

#[derive(Debug, Clone, Copy)]
pub enum SolverType {
    MpcSimplicialCholeskyDefault,
    MpcSupernodalCholeskyDefault,
    MpcSimplicialLuDefault,
}

pub struct Builder<'a> {
    qp: Option<&'a QuadraticProgram>,
    options: SolverOptions,
    type_: Option<SolverType>,
}

impl<'a> Builder<'a> {
    pub fn new() -> Self {
        Self {
            qp: None,
            options: SolverOptions::new(),
            type_: None,
        }
    }

    pub fn with_qp(&mut self, qp: &'a QuadraticProgram) -> &mut Self {
        self.qp = Some(qp);
        self
    }

    pub fn with_options(&mut self, options: &SolverOptions) -> &mut Self {
        self.options = options.clone();
        self
    }

    pub fn with_solver(&mut self, type_: SolverType) -> &mut Self {
        self.type_ = Some(type_);
        self
    }

    pub fn build(&self) -> Result<Box<dyn Solver<Program = QuadraticProgram> + 'a>, Problem> {
        let qp = self.qp.ok_or_else(|| "Quadratic program not set".gloss())?;
        let type_ = self.type_.ok_or_else(|| "Solver type not set".gloss())?;
        let options = &self.options;

        match type_ {
            SolverType::MpcSimplicialCholeskyDefault => {
                Ok(Box::new(PredictorCorrector::<
                    pc::augmented_system::KKTSystem<SimplicialSparseCholesky>,
                    pc::line_search::PrimalDualFeasible,
                    pc::mu_update::MehrotraMuUpdate,
                >::new(qp, options)))
            }
            SolverType::MpcSupernodalCholeskyDefault => {
                Ok(Box::new(PredictorCorrector::<
                    pc::augmented_system::KKTSystem<SupernodalSparseCholesky>,
                    pc::line_search::PrimalDualFeasible,
                    pc::mu_update::MehrotraMuUpdate,
                >::new(qp, options)))
            }
            SolverType::MpcSimplicialLuDefault => Ok(Box::new(PredictorCorrector::<
                pc::augmented_system::KKTSystem<SimplicialSparseLu>,
                pc::line_search::PrimalDualFeasible,
                pc::mu_update::MehrotraMuUpdate,
            >::new(qp, options))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;

    use faer::{ColRef, sparse::Triplet};
    use rstest::fixture;

    use super::*;

    enum Solvers {
        MpcSimplicialCholeskyDefault,
        MpcSupernodalCholeskyDefault,
    }

    #[fixture]
    #[allow(non_snake_case)]
    pub(crate) fn build_simple_qp() -> &'static QuadraticProgram {
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
}
