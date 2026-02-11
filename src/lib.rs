#![feature(const_option_ops)]

use std::any::Any;
use std::ops::Div;

use dyn_clone::DynClone;
use faer::traits::ComplexField;
use faer::traits::num_traits::{Float, PrimInt};
use faer::{Col, ColRef, Index};
use macros::build_options;
use problemo::Problem;

pub trait ElementType: ComplexField + Float + Div<Output = Self> + PrimInt {}
impl<T> ElementType for T where T: ComplexField + Float + Div<Output = T> + PrimInt {}

pub trait IndexType: Copy + PartialEq + Eq + Ord + Index {}
impl<T> IndexType for T where T: Copy + PartialEq + Eq + Ord + Index {}

pub type E = f64;
pub type I = usize;

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

build_options!(name = SolverOptions, registry_name = OPTION_REGISTRY);

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
pub trait IterativeSolver {
    fn get_max_iter(&self) -> usize;

    /// Initialize the solver state.
    fn initialize(&mut self, state: &mut State) {}

    /// Perform a single iteration step.
    fn iterate(&mut self, state: &mut State) -> Result<(), Problem>;

    /// Check if the solver has converged and return the current status.
    fn get_status(&self, state: &State) -> Status;

    /// Run the solver until convergence or maximum iterations.
    fn solve(&mut self, state: &mut State) -> Result<Status, Problem> {
        self.initialize(state);

        let max_iter = self.get_max_iter();
        for iter in 0..max_iter {
            self.iterate(state)?;

            let status = self.get_status(state);
            if status != Status::InProgress {
                println!(
                    "Converged in {} iterations with status: {:?}",
                    iter + 1,
                    status
                );
                return Ok(status);
            }
        }
        println!("Reached maximum iterations without convergence.");
        Ok(Status::IterationLimit)
    }
}

pub struct State {
    x: Col<E>,
    y: Col<E>,
    z_l: Col<E>,
    z_u: Col<E>,

    alpha_primal: E,
    alpha_dual: E,

    primal_infeasibility: E,
    dual_infeasibility: E,
}

impl State {
    pub fn new() -> Self {
        Self {
            x: Col::zeros(0),
            y: Col::zeros(0),
            z_l: Col::zeros(0),
            z_u: Col::zeros(0),

            alpha_primal: E::from(1.),
            alpha_dual: E::from(1.),

            primal_infeasibility: E::from(0.),
            dual_infeasibility: E::from(0.),
        }
    }

    pub fn get_primal(&self) -> ColRef<E> {
        self.x.as_ref()
    }

    pub fn get_dual(&self) -> ColRef<E> {
        self.y.as_ref()
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
}
