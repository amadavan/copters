use macros::{explicit_options, use_option};

use crate::{E, SolverOptions, SolverState};

/// Strategy for computing the step size at each gradient descent iteration.
pub trait StepSize {
    /// Creates a new step size strategy from the solver options.
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Computes the step size for the current iteration.
    fn compute(&mut self, state: &SolverState) -> E;
}

/// Constant step size: `α_k = learning_rate` for all `k`.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "learning_rate", type_ = E, description = "Constant learning rate for gradient descent.")]
pub struct ConstantStepSize {}

impl StepSize for ConstantStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn compute(&mut self, _state: &SolverState) -> E {
        self.options.learning_rate
    }
}

/// Linear decay step size: `α_k = learning_rate / (1 + k)`.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "learning_rate", type_ = E, description = "Initial learning rate for linear decay step size.")]
pub struct LinearDecayStepSize {}

impl StepSize for LinearDecayStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn compute(&mut self, state: &SolverState) -> E {
        self.options.learning_rate / (1. + state.nit as E)
    }
}

/// Quadratic decay step size: `α_k = learning_rate / (1 + k²)`.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "learning_rate", type_ = E, description = "Initial learning rate for quadratic decay step size.")]
pub struct QuadraticDecayStepSize {}

impl StepSize for QuadraticDecayStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn compute(&mut self, state: &SolverState) -> E {
        self.options.learning_rate / (1. + (state.nit as E).powi(2))
    }
}
