#![feature(const_option_ops)]

use std::ops::Div;

use faer::Index;
use faer::traits::ComplexField;
use faer::traits::num_traits::{Float, PrimInt};

extern crate macros;

pub trait ElementType: ComplexField + Float + Div<Output = Self> + PrimInt {}
impl<T> ElementType for T where T: ComplexField + Float + Div<Output = T> + PrimInt {}

pub trait IndexType: Copy + PartialEq + Eq + Ord + Index {}
impl<T> IndexType for T where T: Copy + PartialEq + Eq + Ord + Index {}

pub type E = f64;
pub type I = usize;


pub mod lp;
pub mod nlp;
pub mod stochastic;
pub mod terminators;
pub mod linalg;


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
    fn get_max_iter() -> usize;

    /// Initialize the solver state.
    fn initialize(&mut self) {}

    /// Perform a single iteration step.
    fn iterate(&mut self);

    /// Check if the solver has converged and return the current status.
    fn get_status(&self) -> Status;

    /// Run the solver until convergence or maximum iterations.
    fn solve(&mut self) -> Status {
        self.initialize();

        let max_iter = Self::get_max_iter();
        for iter in 0..max_iter {
            self.iterate();

            let status = self.get_status();
            if status != Status::InProgress {
                println!("Converged in {} iterations with status: {:?}", iter + 1, status);
                return status;
            }
        }
        println!("Reached maximum iterations without convergence.");
        Status::IterationLimit
    }
}