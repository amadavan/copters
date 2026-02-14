use macros::{explicit_options, use_option};

use faer::{Col, unzip, zip};

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

#[explicit_options(name = SolverOptions)]
pub struct BarzilaiBorweinStepSize {
    prev_x: Option<Col<E>>,
    prev_grad: Option<Col<E>>,
}

impl StepSize for BarzilaiBorweinStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            prev_x: None,
            prev_grad: None,

            options: options.into(),
        }
    }

    #[allow(non_snake_case)]
    fn compute(&mut self, state: &SolverState) -> E {
        let step = if let (Some(prev_x), Some(prev_grad)) = (&self.prev_x, &self.prev_grad) {
            let dx: Col<E> = zip!(&state.x, prev_x).map(|unzip!(x_i, x_prev_i)| x_i - x_prev_i);
            let ddL: Col<E> = zip!(state.dL.as_ref().unwrap(), prev_grad)
                .map(|unzip!(g_i, g_prev_i)| g_i - g_prev_i);
            let numerator = zip!(&dx, &ddL)
                .map(|unzip!(dx_i, ddL_i)| dx_i * ddL_i)
                .sum();
            let denominator = zip!(&ddL, &ddL)
                .map(|unzip!(ddL_i, ddL_i2)| ddL_i * ddL_i2)
                .sum();
            if denominator.abs() > 1e-12 {
                numerator / denominator
            } else {
                1.
            }
        } else {
            1.
        };
        self.prev_x = Some(state.x.clone());
        self.prev_grad = state.dL.clone();
        step
    }
}
