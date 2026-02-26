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
    E, I, OptimizationProgram, Solver, SolverHooks, SolverOptions, SolverState, Status,
    linalg::{solver::LinearSolver, vector_ops::cwise_multiply},
    nlp::{
        NLPSolver, NonlinearProgram,
        ipm::{augmented_system::AugmentedSystem, line_search::LineSearch, mu_update::MuUpdate},
    },
};

pub struct Step {
    dx: Col<f64>,
    dy: Col<f64>,
    dz_l: Col<f64>,
    dz_u: Col<f64>,
}

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
    fn iterate(&mut self, state: &mut SolverState) -> Result<(), Problem> {
        // Update hessian
        if let Some(h) = &self.nlp.h {
            state.h = Some((h)(&state.x, &state.y));
        }

        // Update barrier parameter
        let mu = self.mu_update.get(state);

        // Compute search direction
        // Affine scaling direction (predictor step)
        let mut rhs = state.residual.clone();
        let step_aff = self.augmented_system.solve(state, &rhs);

        // Centering step
        let ones = Col::<E>::ones(state.x.nrows());
        rhs.cs_lower += mu * &ones;
        rhs.cs_upper += mu * &ones;

        let step_cen = self.augmented_system.solve(state, &rhs);

        // Corrector step
        rhs.cs_lower -= cwise_multiply(step_aff.dx.as_ref(), step_aff.dz_l.as_ref());
        rhs.cs_upper -= cwise_multiply(step_aff.dx.as_ref(), step_aff.dz_u.as_ref());

        let step_corr = self.augmented_system.solve(state, &rhs);

        // See if we get an acceptable trial point from the line search
        // and iterate till we find one

        // Update the state
        state.residual = self.nlp.compute_residual(state);
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
> Solver for InteriorPointMethod<'a, LinSolve, AS, MU, LS>
{
    fn solve(
        &mut self,
        state: &mut SolverState,
        hooks: &mut SolverHooks,
    ) -> Result<Status, Problem> {
        hooks.callback.init(state);
        // self.initialize(state);
        state.residual = self.nlp.compute_residual(state);

        state.nit = 0;
        state.status = Status::InProgress;

        let max_iter = {
            if self.options.max_iterations > 0 {
                self.options.max_iterations as usize
            } else {
                1e6 as usize // Default to a large number if not set
            }
        };
        for iter in 0..max_iter {
            state.nit = iter;
            self.iterate(state)?;

            let status = state.status;
            if status != Status::InProgress {
                println!(
                    "Converged in {} iterations with status: {:?}",
                    iter + 1,
                    status
                );
                return Ok(status);
            }

            hooks.callback.call(state);
            if let Some(terminator_status) = hooks.terminator.terminate(state) {
                println!(
                    "Terminated in {} iterations with status: {:?}",
                    iter + 1,
                    terminator_status
                );
                return Ok(terminator_status);
            }
        }
        println!("Reached maximum iterations without convergence.");
        Ok(Status::IterationLimit)
    }
}
