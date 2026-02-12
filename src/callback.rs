use macros::build_option_enum;

use crate::{SolverOptions, SolverState};

pub trait Callback {
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    fn call(&mut self, state: &SolverState);
}

pub struct NoOpCallback {}

impl Callback for NoOpCallback {
    fn new(_options: &SolverOptions) -> Self {
        Self {}
    }

    fn call(&mut self, _state: &SolverState) {
        // Do nothing
    }
}

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
