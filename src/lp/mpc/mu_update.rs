use macros::{explicit_options, use_option};

use crate::{
    E, SolverOptions, SolverState, linalg::vector_ops::cwise_multiply_finite, lp::LinearProgram,
};

pub trait MuUpdate<'a> {
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;
    fn get(&mut self, state: &SolverState) -> E;
}

#[use_option(name = "mu_fixed", type_ = E, default = "1.", description = "Constant value for the barrier parameter mu")]
pub struct ConstantMuUpdate<'a> {
    lp: &'a LinearProgram,
    mu: E,
}

impl<'a> MuUpdate<'a> for ConstantMuUpdate<'a> {
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self {
        Self {
            lp,
            mu: options.get_option::<E>("mu_fixed").unwrap(), // Default constant value for mu
        }
    }

    fn get(&mut self, _state: &SolverState) -> E {
        self.mu
    }
}

#[explicit_options(name = SolverOptions)]
#[use_option(name = "mu_min", type_ = f64, default = "1e-7", description = "Minimum value for the barrier parameter mu")]
#[use_option(name = "mu_max", type_ = f64, default = "1e7", description = "Maximum value for the barrier parameter mu")]
pub struct AdaptiveMuUpdate<'a> {
    lp: &'a LinearProgram,
    mu: f64,
}

impl<'a> MuUpdate<'a> for AdaptiveMuUpdate<'a> {
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self {
        Self {
            lp,
            mu: 1.,
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
        self.mu = (l + u) / state.x.nrows() as E;

        self.mu
    }
}
