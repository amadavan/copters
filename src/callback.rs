use macros::build_option_enum;

use crate::{SolverOptions, SolverState};

/// Hook invoked once per solver iteration for logging, monitoring, or early stopping.
pub trait Callback {
    /// Creates a new callback from solver options.
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Called at the end of each iteration with the current solver state.
    fn call(&mut self, state: &SolverState);
}

/// A callback that does nothing. Use when no per-iteration output is needed.
pub struct NoOpCallback {}

impl Callback for NoOpCallback {
    fn new(_options: &SolverOptions) -> Self {
        Self {}
    }

    fn call(&mut self, _state: &SolverState) {
        // Do nothing
    }
}

/// Prints primal and dual infeasibility to stdout each iteration.
pub struct ConvergenceOutput {}

impl Callback for ConvergenceOutput {
    fn new(_options: &SolverOptions) -> Self {
        Self {}
    }

    fn call(&mut self, state: &SolverState) {
        println!(
            "Primal Infeasibility: {}, Dual Infeasibility: {}",
            state.get_primal_infeasibility(),
            state.get_dual_infeasibility()
        );
    }
}

build_option_enum!(
    trait_ = Callback,
    name = "Callbacks",
    variants = (NoOpCallback, ConvergenceOutput),
    new_arguments = (&SolverOptions,),
    doc_header = "An enum representing different callbacks for the optimization solver. Each variant corresponds to a specific callback strategy."
);
