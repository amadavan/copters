#![feature(const_option_ops)]

use std::any::Any;
use std::ops::Div;

use dyn_clone::DynClone;
use faer::traits::ComplexField;
use faer::traits::num_traits::{Float, PrimInt};
use faer::{Col, Index};
use macros::{build_options, use_option};
use problemo::Problem;

pub trait ElementType: ComplexField + Float + Div<Output = Self> + PrimInt {}
impl<T> ElementType for T where T: ComplexField + Float + Div<Output = T> + PrimInt {}

pub trait IndexType: Copy + PartialEq + Eq + Ord + Index {}
impl<T> IndexType for T where T: Copy + PartialEq + Eq + Ord + Index {}

pub type E = f64;
pub type I = usize;

pub mod callback;
pub mod interface;
pub mod linalg;
pub mod lp;
pub mod nlp;
pub mod stochastic;
pub mod terminators;

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
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Status {
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

/// Trait for iterative optimization solvers.
///
/// Provides a standard interface for algorithms that proceed by repeated iteration,
/// such as simplex, interior-point, or gradient-based methods.
pub trait Solver {
    /// Run the solver until convergence or maximum iterations.
    fn solve(
        &mut self,
        state: &mut SolverState,
        properties: &mut Properties,
    ) -> Result<Status, Problem>;
}

#[derive(Debug, Clone)]
pub struct SolverState {
    status: Status,
    nit: usize,

    x: Col<E>,
    y: Col<E>,
    z_l: Col<E>,
    z_u: Col<E>,

    alpha_primal: E,
    alpha_dual: E,

    primal_infeasibility: E,
    dual_infeasibility: E,
    complimentary_slack_lower: E,
    complimentary_slack_upper: E,

    // Solver-specific state can be added here as needed
    sigma: E,
    mu: E,
    safety_factor: E,
}

impl SolverState {
    pub fn new(x: Col<E>, y: Col<E>, z_l: Col<E>, z_u: Col<E>) -> Self {
        Self {
            status: Status::InProgress,
            nit: 0,

            x,
            y,
            z_l,
            z_u,

            alpha_primal: E::from(1.),
            alpha_dual: E::from(1.),

            primal_infeasibility: E::from(0.),
            dual_infeasibility: E::from(0.),
            complimentary_slack_lower: E::from(0.),
            complimentary_slack_upper: E::from(0.),

            sigma: E::from(1.),
            mu: E::from(1.),
            safety_factor: E::from(1.),
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

    pub fn get_primal_infeasibility(&self) -> E {
        self.primal_infeasibility
    }

    pub fn get_dual_infeasibility(&self) -> E {
        self.dual_infeasibility
    }

    pub fn get_complimentary_slack_lower(&self) -> E {
        self.complimentary_slack_lower
    }

    pub fn get_complimentary_slack_upper(&self) -> E {
        self.complimentary_slack_upper
    }

    pub(crate) fn get_sigma_mu(&self) -> (E, E) {
        (self.sigma, self.mu)
    }

    pub(crate) fn set_sigma_mu(&mut self, sigma: E, mu: E) {
        self.sigma = sigma;
        self.mu = mu;
    }

    pub(crate) fn set_safety_factor(&mut self, safety_factor: E) {
        self.safety_factor = safety_factor;
    }
}

#[use_option(name="Callback", type_=crate::callback::Callbacks, description="Callback for the solver.")]
#[use_option(name="Terminator", type_=crate::terminators::Terminators, default="NullTerminator", description="Terminator for the solver.")]
pub struct Properties {
    callback: Box<dyn crate::callback::Callback>,
    terminator: Box<dyn crate::terminators::Terminator>,
}

build_options!(name = SolverOptions, registry_name = OPTION_REGISTRY);
