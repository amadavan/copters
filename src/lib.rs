#![feature(const_option_ops)]
#![feature(stmt_expr_attributes)]

use std::any::Any;
use std::ops::Div;

use derive_more::PartialEq;
use dyn_clone::{DynClone, clone_box};
use faer::sparse::SparseColMat;
use faer::traits::ComplexField;
use faer::traits::num_traits::{Float, PrimInt};
use faer::{Col, Index};
use macros::build_options;
use problemo::Problem;

use crate::callback::Callback;

pub trait ElementType: ComplexField + Float + Div<Output = Self> + PrimInt {}
impl<T> ElementType for T where T: ComplexField + Float + Div<Output = T> + PrimInt {}

pub trait IndexType: Copy + PartialEq + Eq + Ord + Index {}
impl<T> IndexType for T where T: Copy + PartialEq + Eq + Ord + Index {}

pub type E = f64;
pub type I = usize;

pub mod callback;
pub mod interface;
pub(crate) mod ipm;
pub mod linalg;
pub mod lp;
pub mod nlp;
pub mod qp;
pub mod stochastic;
pub mod terminators;
pub mod utils;

#[cfg(feature = "data-loaders")]
pub mod data_loaders;

#[cfg(test)]
pub mod tests;

pub trait OptionTrait: Any + Sync + Send + DynClone {}
impl OptionTrait for &'static str {}
impl OptionTrait for String {}
impl OptionTrait for bool {}
impl OptionTrait for usize {}
impl OptionTrait for u8 {}
impl OptionTrait for u16 {}
impl OptionTrait for u32 {}
impl OptionTrait for u64 {}
impl OptionTrait for i8 {}
impl OptionTrait for i16 {}
impl OptionTrait for i32 {}
impl OptionTrait for i64 {}
impl OptionTrait for f32 {}
impl OptionTrait for f64 {}

impl Clone for Box<dyn OptionTrait> {
    fn clone(&self) -> Self {
        dyn_clone::clone_box(&**self)
    }
}

/// Status codes for optimization solvers.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Status {
    #[default]
    /// The solver is still running.
    InProgress,
    /// An optimal solution was found.
    Optimal,
    /// The problem is infeasible.
    Infeasible,
    /// The problem is unbounded.
    Unbounded,
    /// The status is unknown or not determined.
    Unknown,
    /// The solver stopped due to a time limit.
    TimeLimit,
    /// The solver stopped due to an iteration limit.
    IterationLimit,
    /// The solver was interrupted (e.g., by user or signal).
    Interrupted,
}

pub trait OptimizationProgram {
    fn update_residual(&self, state: &mut SolverState);
}

/// Trait for iterative optimization solvers.
///
/// Provides a standard interface for algorithms that proceed by repeated iteration,
/// such as simplex, interior-point, or gradient-based methods.
pub trait IterativeSolver {
    fn get_program(&self) -> &dyn OptimizationProgram;

    fn get_max_iterations(&self) -> usize;

    fn initialize(&mut self, _state: &mut SolverState) {
        // Default implementation does nothing, but can be overridden by specific solvers
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<Status, Problem>;

    fn solve(
        &mut self,
        state: &mut SolverState,
        hooks: &mut SolverHooks,
    ) -> Result<Status, Problem> {
        hooks.callback.init(state);

        self.initialize(state);

        state.nit = 0;
        state.status = Status::InProgress;

        let max_iter = {
            let max_iter = self.get_max_iterations();
            if max_iter > 0 {
                max_iter as usize
            } else {
                1000
            }
        };

        for iter in 0..max_iter {
            state.nit = iter;
            self.iterate(state)?;

            let status = state.status;
            if status != Status::InProgress {
                println!(
                    "Converged in {} iterations with status: {:?}",
                    iter + 1,
                    status
                );
                return Ok(status);
            }

            hooks.callback.call(state);
            if let Some(terminator_status) = hooks.terminator.terminate(state) {
                println!(
                    "Terminated in {} iterations with status: {:?}",
                    iter + 1,
                    terminator_status
                );
                return Ok(terminator_status);
            }
        }
        println!("Reached maximum iterations without convergence.");
        Ok(Status::IterationLimit)
    }
}

#[derive(Debug, Clone)]
#[allow(non_snake_case, unused)]
pub struct SolverState {
    status: Status,
    nit: usize,

    // Primal-Dual Variables
    x: Col<E>,
    y: Col<E>,
    z_l: Col<E>,
    z_u: Col<E>,

    // Residual
    dual_feasibility: Col<E>,
    primal_feasibility: Col<E>,
    cs_lower: Col<E>,
    cs_upper: Col<E>,

    // Step size
    alpha_primal: E,
    alpha_dual: E,

    // IPM-specific state
    sigma: Option<E>,
    mu: Option<E>,
    tau: Option<E>,
    safety_factor: Option<E>,

    // NLP-specific state
    f: Option<E>,
    g: Option<Col<E>>,
    df: Option<Col<E>>,
    dg: Option<SparseColMat<I, E>>,
    h: Option<SparseColMat<I, E>>,
    dL: Option<Col<E>>,
}

impl SolverState {
    pub fn new(x: Col<E>, y: Col<E>, z_l: Col<E>, z_u: Col<E>) -> Self {
        Self {
            status: Status::InProgress,
            nit: 0,

            x: x.clone(),
            y: y.clone(),
            z_l: z_l.clone(),
            z_u: z_u.clone(),

            alpha_primal: E::from(1.),
            alpha_dual: E::from(1.),

            dual_feasibility: Col::<E>::zeros(x.nrows()),
            primal_feasibility: Col::<E>::zeros(y.nrows()),
            cs_lower: Col::<E>::zeros(z_l.nrows()),
            cs_upper: Col::<E>::zeros(z_u.nrows()),

            sigma: None,
            mu: None,
            tau: None,
            safety_factor: None,

            f: None,
            g: None,
            df: None,
            dg: None,
            h: None,
            dL: None,
        }
    }

    pub fn get_status(&self) -> Status {
        self.status
    }

    pub fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    pub fn get_primal(&self) -> &Col<E> {
        &self.x
    }

    pub fn get_dual(&self) -> &Col<E> {
        &self.y
    }

    pub fn get_reduced_cost(&self) -> Col<E> {
        &self.z_l - &self.z_u
    }

    pub fn get_dual_feasibility(&self) -> &Col<E> {
        &self.dual_feasibility
    }

    pub fn get_primal_feasibility(&self) -> &Col<E> {
        &self.primal_feasibility
    }

    pub fn get_cs_lower(&self) -> &Col<E> {
        &self.cs_lower
    }

    pub fn get_cs_upper(&self) -> &Col<E> {
        &self.cs_upper
    }
}

pub struct SearchDirection {
    dx: Col<E>,
    dy: Col<E>,
    dz_l: Col<E>,
    dz_u: Col<E>,
}

impl SearchDirection {
    pub fn get_dx(&self) -> &Col<E> {
        &self.dx
    }

    pub fn get_dy(&self) -> &Col<E> {
        &self.dy
    }

    pub fn get_dz_l(&self) -> &Col<E> {
        &self.dz_l
    }

    pub fn get_dz_u(&self) -> &Col<E> {
        &self.dz_u
    }
}

pub struct SolverHooks {
    callback: Box<dyn Callback>,
    terminator: Box<dyn crate::terminators::Terminator>,
}

impl Clone for SolverHooks {
    fn clone(&self) -> Self {
        Self {
            callback: clone_box(&*self.callback),
            terminator: clone_box(&*self.terminator),
        }
    }
}

build_options!(name = SolverOptions, registry_name = OPTION_REGISTRY);
