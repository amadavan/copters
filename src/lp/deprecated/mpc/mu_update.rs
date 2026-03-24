use std::marker::PhantomData;

use macros::{explicit_options, use_option};

use crate::{
    E, SolverOptions, SolverState, linalg::vector_ops::cwise_multiply_finite, lp::LinearProgram,
};

/// Strategy for computing the barrier parameter `mu`.
///
/// The barrier parameter controls the trade-off between optimality and
/// centrality in an interior-point method. Implementations determine
/// how `mu` evolves across iterations.
pub trait MuUpdate<'a> {
    /// Creates a new instance from the linear program and solver options.
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;
    /// Returns the current value of `mu` for the given solver state.
    fn get(&mut self, state: &SolverState) -> E;
}

/// Returns a fixed barrier parameter value across all iterations.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "mu_fixed", type_ = E, default = "1.", description = "Constant value for the barrier parameter mu")]
pub struct ConstantMuUpdate<'a> {
    _a: PhantomData<&'a ()>,
}

impl<'a> MuUpdate<'a> for ConstantMuUpdate<'a> {
    fn new(_lp: &'a LinearProgram, options: &SolverOptions) -> Self {
        Self {
            _a: PhantomData,
            options: options.into(),
        }
    }

    fn get(&mut self, _state: &SolverState) -> E {
        self.options.mu_fixed
    }
}

/// Computes `mu` from the complementarity conditions of the current iterate:
///
/// ```text
/// mu = ((x - l)^T z_l + (x - u)^T z_u) / n
/// ```
///
/// Infinite bounds are ignored (treated as zero contribution).
#[explicit_options(name = SolverOptions)]
#[use_option(name = "mu_min", type_ = E, default = "1e-7", description = "Minimum value for the barrier parameter mu")]
#[use_option(name = "mu_max", type_ = E, default = "1e7", description = "Maximum value for the barrier parameter mu")]
pub struct AdaptiveMuUpdate<'a> {
    lp: &'a LinearProgram,
}

impl<'a> MuUpdate<'a> for AdaptiveMuUpdate<'a> {
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self {
        Self {
            lp,
            options: options.into(),
        }
    }

    fn get(&mut self, state: &SolverState) -> E {
        // (x-l) where 0 if inf
        // (x-u) where 0 if inf
        // (x-l)^T z_lower
        // (x-u)^T z_upper
        // sum / n

        let xl = &state.x - &self.lp.l;
        let xu = &state.x - &self.lp.u;

        let l = cwise_multiply_finite(state.z_l.as_ref(), xl.as_ref()).sum();
        let u = cwise_multiply_finite(state.z_u.as_ref(), xu.as_ref()).sum();
        let mu = (l + u) / state.x.nrows() as E;

        mu.clamp(self.options.mu_min, self.options.mu_max)
    }
}
