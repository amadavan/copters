use macros::use_option;

use crate::{E, SolverState};

pub trait MuOracle {
    fn compute_mu(&mut self, state: &SolverState) -> E;
}

#[use_option(name = "mu_max", type_ = E, description = "Maximum value for the barrier parameter mu.")]
#[use_option(name = "mu_min", type_ = E, description = "Minimum value for the barrier parameter mu.")]
pub struct LoquMuOracle;

impl MuOracle for LoquMuOracle {
    fn compute_mu(&mut self, state: &SolverState) -> E {
        // Implement the logic to compute mu based on the current state
        // This could involve using the complementarity conditions, residuals, or other metrics

        // Placeholder implementation
        1e-3 // Return a dummy value for now
    }
}

pub struct ProbingMuOracle;

impl MuOracle for ProbingMuOracle {
    fn compute_mu(&mut self, state: &SolverState) -> E {
        // Implement the logic to compute mu based on the current state
        // This could involve using the complementarity conditions, residuals, or other metrics

        // Placeholder implementation
        1e-3 // Return a dummy value for now
    }
}

pub struct QualityFunctionMuOracle;

impl MuOracle for QualityFunctionMuOracle {
    fn compute_mu(&mut self, state: &SolverState) -> E {
        // Implement the logic to compute mu based on the current state
        // This could involve using the complementarity conditions, residuals, or other metrics

        // Placeholder implementation
        1e-3 // Return a dummy value for now
    }
}
