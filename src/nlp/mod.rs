pub mod gd;
pub mod ipm;

use std::str::FromStr;

use faer::{Col, sparse::SparseColMat};
use macros::use_option;
use problemo::{Problem, common::IntoCommonProblem};

use crate::{
    E, I, OptimizationProgram, OptionTrait, Solver, SolverOptions, SolverState,
    linalg::{cholesky::SimplicialSparseCholesky, vector_ops::cwise_multiply_finite},
    nlp::ipm::{
        augmented_system::{AugmentedSystem, StandardSystem},
        line_search::PDFeasibileLineSearch,
        mu_update::AdaptiveMuUpdate,
    },
};

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
#[use_option(name = "nlp_solver_type", type_ = crate::nlp::NLPSolverType, default = "gradient_descent", description = "Type of NLP solver to use.")]
pub struct NonlinearProgram {
    /// Number of decision variables.
    n_var: I,
    /// Number of equality constraints.
    n_cons: I,

    /// Objective function `f(x) -> scalar`.
    f: Box<dyn Fn(&Col<E>) -> E>,
    /// Equality constraint function `g(x) -> Col`.
    g: Box<dyn Fn(&Col<E>) -> Col<E>>,
    /// Gradient of the objective `∇f(x)`.
    df: Box<dyn Fn(&Col<E>) -> Col<E>>,
    /// Jacobian of the constraints `∇g(x)` (sparse).
    dg: Box<dyn Fn(&Col<E>) -> SparseColMat<I, E>>,
    /// Hessian of the Lagrangian `∇²L(x, y)` (optional, sparse).
    h: Option<Box<dyn Fn(&Col<E>, &Col<E>) -> SparseColMat<I, E>>>,

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
        let h = {
            if let Some(h_fn) = h {
                Some(Box::new(h_fn) as Box<dyn Fn(&Col<E>, &Col<E>) -> SparseColMat<I, E>>)
            } else {
                None
            }
        };

        Self {
            n_var,
            n_cons,
            f: Box::new(f),
            g: Box::new(g),
            df: Box::new(df),
            dg: Box::new(dg),
            h,
            l,
            u,
        }
    }

    pub fn new_boxed(
        n_var: I,
        n_cons: I,
        f: Box<dyn Fn(&Col<E>) -> E>,
        g: Box<dyn Fn(&Col<E>) -> Col<E>>,
        df: Box<dyn Fn(&Col<E>) -> Col<E>>,
        dg: Box<dyn Fn(&Col<E>) -> SparseColMat<I, E>>,
        h: Option<Box<dyn Fn(&Col<E>, &Col<E>) -> SparseColMat<I, E>>>,
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

    pub fn solver_builder<'a>(&'a self) -> NLPSolverBuilder<'a> {
        NLPSolverBuilder::new().with_nlp(self)
    }
}

impl OptimizationProgram for NonlinearProgram {
    fn update_residual(&self, state: &mut SolverState) {
        let (x, y, z_l, z_u) = (&state.x, &state.y, &state.z_l, &state.z_u);
        let zero = Col::<E>::zeros(self.n_var);
        let inf = E::INFINITY * Col::<E>::ones(self.n_var);
        let (l, u) = (self.l().unwrap_or(&zero), self.u().unwrap_or(&inf));

        state.dual_feasibility = -self.df(x) + self.dg(x).transpose() * y + z_l + z_u;
        state.primal_feasibility = self.g(x);
        state.cs_lower = -cwise_multiply_finite(z_l.as_ref(), (x - l).as_ref());
        state.cs_upper = -cwise_multiply_finite(z_u.as_ref(), (x - u).as_ref());
    }
}

pub trait NLPSolver<'a>: IterativeSolver {
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;
    // Define any additional methods specific to NLP solvers here.
}

#[derive(Copy, Clone, Debug, Default)]
pub enum NLPSolverType {
    #[default]
    GradientDescent,
    InteriorPointMethod,
}

impl OptionTrait for NLPSolverType {}

impl FromStr for NLPSolverType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "interior_point_method" | "ipm" => Ok(NLPSolverType::InteriorPointMethod),
            "gradient_descent" | "gd" => Ok(NLPSolverType::GradientDescent),
            _ => Err(format!("Invalid NLP solver type: {}", s)),
        }
    }
}

pub struct NLPSolverBuilder<'a> {
    nlp: Option<&'a NonlinearProgram>,
    solver_type: Option<NLPSolverType>,
    options: SolverOptions,
    // Add any additional configuration options here.
}

impl<'a> NLPSolverBuilder<'a> {
    pub fn new() -> Self {
        Self {
            nlp: None,
            solver_type: None,
            options: SolverOptions::new(),
        }
    }

    pub fn with_nlp(mut self, nlp: &'a NonlinearProgram) -> Self {
        self.nlp = Some(nlp);
        self
    }

    pub fn with_options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_solver(mut self, solver_type: NLPSolverType) -> Self {
        self.solver_type = Some(solver_type);
        self
    }

    // Add any additional builder methods here.

    pub fn build(self) -> Result<Box<dyn NLPSolver<'a> + 'a>, Problem> {
        // Get the nonlinear program from the builder
        let nlp = self
            .nlp
            .ok_or_else(|| "Nonlinear program must be provided".gloss())?;

        // Get the solver type from the builder or fallback to options
        let solver_type = self
            .solver_type
            .or(self.options.get_option::<NLPSolverType>("nlp_solver_type"))
            .ok_or_else(|| "Solver type must be specified".gloss())?;

        // Construct the appropriate solver based on the solver type
        match solver_type {
            NLPSolverType::InteriorPointMethod => {
                Ok(Box::new(ipm::InteriorPointMethod::<
                    SimplicialSparseCholesky,
                    StandardSystem<SimplicialSparseCholesky>,
                    AdaptiveMuUpdate,
                    PDFeasibileLineSearch,
                >::new(nlp, &self.options)))
            }
            NLPSolverType::GradientDescent => Ok(Box::new(gd::GradientDescent::<
                gd::stepsize::ConstantStepSize,
            >::new(nlp, &self.options))),
            _ => Err("Invalid solver type.".gloss()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::tests::build_simple_lp;
    use crate::{
        I, SolverHooks, Status, callback::ConvergenceOutput, terminators::ConvergenceTerminator,
    };
    use faer::Col;
    use rstest::rstest;

    #[rstest]
    fn test_simple_lp(#[values(NLPSolverType::InteriorPointMethod)] solver_type: NLPSolverType) {
        let lp = build_simple_lp();
        let nlp: NonlinearProgram = lp.into();

        let mut state = SolverState::new(
            Col::ones(lp.get_objective().nrows()),
            Col::ones(lp.get_rhs().nrows()),
            Col::ones(lp.get_objective().nrows()),
            -Col::<E>::ones(lp.get_objective().nrows()),
        );

        let mut options = SolverOptions::new();
        options.set_option("max_iterations", 10 as I).unwrap();

        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut solver = NonlinearProgram::solver_builder(&nlp)
            .with_solver(solver_type)
            .with_options(options.clone())
            .build()
            .unwrap();
        let status = solver.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }

    #[rstest]
    fn test_simple_qp(#[values(NLPSolverType::InteriorPointMethod)] solver_type: NLPSolverType) {
        let qp = build_simple_lp();
        let nlp: NonlinearProgram = qp.into();

        let mut state = SolverState::new(
            Col::ones(qp.get_n_vars()),
            Col::ones(qp.get_n_cons()),
            Col::ones(qp.get_n_vars()),
            -Col::<E>::ones(qp.get_n_vars()),
        );

        let mut options = SolverOptions::new();
        options.set_option("max_iterations", 10 as I).unwrap();

        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut solver = NonlinearProgram::solver_builder(&nlp)
            .with_solver(solver_type)
            .with_options(options.clone())
            .build()
            .unwrap();
        let status = solver.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }
}
