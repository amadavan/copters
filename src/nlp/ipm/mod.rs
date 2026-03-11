//! # Interior Point Method (IPM) for Nonlinear Programming
//!
//! This module implements a primal-dual interior point method for solving
//! nonlinear programming problems of the form:
//!
//! ```text
//!   min  f(x)
//!   s.t. g(x) = 0
//!        x_L <= x <= x_U
//! ```
//!
//! ## Acknowledgment
//!
//! This implementation is **heavily inspired by
//! [IPOPT](https://github.com/coin-or/Ipopt)** (Interior Point OPTimizer),
//! developed by Andreas Wächter and Carl Laird as part of the
//! [COIN-OR](https://www.coin-or.org/) initiative. The algorithmic design,
//! including the barrier parameter update strategy, the augmented system
//! formulation, and the filter line search procedure, closely follows the
//! approach described in:
//!
//! > A. Wächter and L. T. Biegler, "On the Implementation of a Primal-Dual
//! > Interior Point Filter Line Search Algorithm for Large-Scale Nonlinear
//! > Programming", *Mathematical Programming* 106(1), pp. 25-57, 2006.
//!
//! ## License Compatibility
//!
//! IPOPT is released under the **Eclipse Public License 2.0 (EPL-2.0)**. This
//! module is an independent implementation written from scratch in Rust and does
//! not contain any source code from IPOPT. It is licensed under the MIT License
//! along with the rest of this crate. The algorithmic ideas and mathematical
//! formulations used here are not subject to copyright.

pub mod augmented_system;
pub mod line_search;
pub mod mu_oracle;
pub mod mu_update;

use std::marker::PhantomData;

use faer::Col;
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, I, IterativeSolver, OptimizationProgram, SolverOptions, SolverState, Status,
    ipm::RHS,
    linalg::{solver::LinearSolver, vector_ops::cwise_multiply_finite},
    nlp::{
        NLPSolver, NonlinearProgram,
        ipm::{augmented_system::AugmentedSystem, line_search::LineSearch, mu_update::MuUpdate},
    },
};

#[explicit_options(name = SolverOptions)]
#[use_option(name = "max_iterations", type_ = I, description = "Maximum number of iterations for the interior point method")]
pub struct InteriorPointMethod<
    'a,
    LinSolve: LinearSolver,
    AS: AugmentedSystem<'a, LinSolve>,
    MU: MuUpdate,
    LS: LineSearch<'a>,
> {
    nlp: &'a NonlinearProgram,
    mu_update: MU,
    augmented_system: AS,
    line_search: LS,

    _lin_solve: PhantomData<LinSolve>,
}

impl<
    'a,
    LinSolve: LinearSolver,
    AS: AugmentedSystem<'a, LinSolve>,
    MU: MuUpdate,
    LS: LineSearch<'a>,
> InteriorPointMethod<'a, LinSolve, AS, MU, LS>
{
    fn initialize(&mut self, _state: &mut SolverState) {}

    fn iterate(&mut self, state: &mut SolverState) -> Result<(), Problem> {
        // Update function evaluations and derivatives at the current iterate
        state.df = Some(self.nlp.df(&state.x));
        state.g = Some(self.nlp.g(&state.x));
        state.dg = Some(self.nlp.dg(&state.x));

        // Update hessian
        if let Some(h) = &self.nlp.h {
            state.h = Some((h)(&state.x, &state.y));
        }

        // Update barrier parameter
        state.mu = Some(self.mu_update.get(state));

        // Compute search direction
        // Affine scaling direction (predictor step)
        let mut rhs = RHS::from(&*state);
        let step_aff = self.augmented_system.solve(state, &rhs)?;

        state.safety_factor = Some(1.);
        let alpha_aff = self.line_search.perform_line_search(state, &step_aff);

        let mut state_aff = state.clone();
        state_aff.x += alpha_aff * &step_aff.dx;
        state_aff.y += alpha_aff * &step_aff.dy;
        state_aff.z_l += alpha_aff * &step_aff.dz_l;
        state_aff.z_u += alpha_aff * &step_aff.dz_u;

        let mu_aff = self.mu_update.get(&state_aff);
        let sigma = E::powf(mu_aff / state.mu.unwrap_or(E::from(1.)), 3.);
        state.mu = Some(sigma * state.mu.unwrap());

        // Centering step
        let ones = Col::<E>::ones(state.x.nrows());
        *rhs.r_l_mut() += state.mu.unwrap() * &ones;
        *rhs.r_u_mut() += state.mu.unwrap() * &ones;

        // let step_cen = self.augmented_system.solve(state, &rhs)?;

        // Corrector step
        *rhs.r_l_mut() -=
            cwise_multiply_finite(step_aff.get_dz_l().as_ref(), step_aff.get_dx().as_ref());
        *rhs.r_u_mut() -=
            cwise_multiply_finite(step_aff.get_dz_u().as_ref(), step_aff.get_dx().as_ref());

        let step_corr = self.augmented_system.solve(state, &rhs)?;

        state.safety_factor = Some(0.99); // Reduce step length to maintain stability
        let alpha_corr = self.line_search.perform_line_search(state, &step_corr);

        // See if we get an acceptable trial point from the line search
        // and iterate till we find one

        // Update the state with the corrector step and step lengths
        state.x += alpha_corr * &step_corr.dx;
        state.y += alpha_corr * &step_corr.dy;
        state.z_l += alpha_corr * &step_corr.dz_l;
        state.z_u += alpha_corr * &step_corr.dz_u;
        state.alpha_primal = alpha_corr;
        state.alpha_dual = alpha_corr;

        // Update the state
        self.nlp.update_residual(state);
        state.status = Status::InProgress;

        Ok(())
    }
}

impl<
    'a,
    LinSolve: LinearSolver,
    AS: AugmentedSystem<'a, LinSolve>,
    MU: MuUpdate,
    LS: LineSearch<'a>,
> NLPSolver<'a> for InteriorPointMethod<'a, LinSolve, AS, MU, LS>
{
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self {
        Self {
            nlp,
            line_search: LS::new(nlp, options),
            mu_update: MU::new(options),
            augmented_system: AS::new(nlp, options),

            options: options.into(),

            _lin_solve: PhantomData,
        }
    }
}

impl<
    'a,
    LinSolve: LinearSolver,
    AS: AugmentedSystem<'a, LinSolve>,
    MU: MuUpdate,
    LS: LineSearch<'a>,
> IterativeSolver for InteriorPointMethod<'a, LinSolve, AS, MU, LS>
{
    fn get_max_iterations(&self) -> usize {
        if self.options.max_iterations > 0 {
            self.options.max_iterations as usize
        } else {
            1e2 as usize // Default to a large number if not set
        }
    }

    fn get_program(&self) -> &dyn OptimizationProgram {
        self.nlp
    }

    fn initialize(&mut self, state: &mut SolverState) {
        self.initialize(state);
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<Status, Problem> {
        self.iterate(state)?;
        Ok(state.status)
    }
}
