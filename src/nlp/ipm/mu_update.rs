use macros::explicit_options;

use crate::{E, SolverOptions, SolverState};

pub trait MuUpdate {
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Updates the barrier parameter `mu` based on the current state of the solver.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state of the solver, containing the current iterate and other relevant information.
    ///
    /// # Returns
    ///
    /// The updated value of the barrier parameter `mu`.
    fn get(&mut self, state: &SolverState) -> E;
}

#[explicit_options(name = SolverOptions)]
pub struct AdaptiveMuUpdate {
    // Add any necessary fields for the adaptive mu update strategy
}

impl MuUpdate for AdaptiveMuUpdate {
    fn new(options: &SolverOptions) -> Self {
        // Initialize any necessary fields based on the solver options
        Self {
            options: options.into(),
        }
    }

    fn get(&mut self, state: &SolverState) -> E {
        // Implement the logic to compute the new value of mu based on the current state
        // This could involve using the complementarity conditions, residuals, or other metrics

        // TODO: handle different modes, Fix/Free, etc.

        // Placeholder implementation
        1e-3 // Return a dummy value for now
    }
}
