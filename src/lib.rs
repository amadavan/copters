#![feature(const_option_ops)]
#![feature(stmt_expr_attributes)]

use std::any::Any;
use std::ops::Div;

use dyn_clone::{DynClone, clone_box};
use faer::Index;
use faer::traits::ComplexField;
use faer::traits::num_traits::{Float, PrimInt};
use macros::build_options;
use problemo::Problem;

use crate::callback::Callback;
use crate::state::{SolverState, Status};

pub trait ElementType: ComplexField + Float + Div<Output = Self> + PrimInt {}
impl<T> ElementType for T where T: ComplexField + Float + Div<Output = T> + PrimInt {}

pub trait IndexType: Copy + PartialEq + Eq + Ord + Index {}
impl<T> IndexType for T where T: Copy + PartialEq + Eq + Ord + Index {}

pub type E = f64;
pub type I = usize;

pub mod callback;
// pub mod interface;
// pub(crate) mod ipm;
pub mod linalg;
pub mod lp;
pub mod nlp;
// pub mod qp;
pub mod state;
// pub mod stochastic;
pub mod model;
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

        state.set_nit(0);
        state.set_status(Status::InProgress);

        let max_iter = {
            let max_iter = self.get_max_iterations();
            if max_iter > 0 {
                max_iter as usize
            } else {
                1000
            }
        };

        for iter in 0..max_iter {
            state.inc_nit();
            self.iterate(state)?;

            let status = state.status();
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
