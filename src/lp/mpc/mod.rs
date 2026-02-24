use std::marker::PhantomData;

use faer::{Col, traits::num_traits::pow};
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, I, OptimizationProgram, Solver, SolverHooks, SolverOptions, SolverState, Status,
    linalg::{solver::LinearSolver, vector_ops::cwise_multiply_finite},
    lp::{
        LPSolver, LinearProgram,
        mpc::{augmented_system::AugmentedSystem, mu_update::MuUpdate},
    },
};

pub mod augmented_system;
pub mod line_search;
pub mod mu_update;

/// A primal-dual search direction `(dx, dy, dz_l, dz_u)`.
pub struct Step {
    dx: Col<E>,
    dy: Col<E>,
    dz_l: Col<E>,
    dz_u: Col<E>,
}

/// Mehrotra predictor-corrector interior-point solver for linear programs.
///
/// Each iteration performs two solves of the augmented system:
/// 1. **Affine (predictor) step** — a pure Newton step toward the boundary.
/// 2. **Corrector step** — adjusts centering parameter `sigma` based on the
///    affine step and adds second-order corrections to the complementarity.
///
/// The solver is generic over the linear system factorization (`Solver`),
/// augmented system formulation (`System`), barrier parameter strategy (`MU`),
/// and line search (`LS`).
#[explicit_options(name = SolverOptions)]
#[use_option(name = "max_iterations", type_=I, default="0", description="Maximum number of iterations (0 uses solver defaults).")]
pub struct MehrotraPredictorCorrector<
    'a,
    LinSolve: LinearSolver,
    Sys: AugmentedSystem<'a, LinSolve>,
    MU: MuUpdate<'a>,
> {
    lp: &'a LinearProgram,

    system: Sys,
    mu_updater: MU,

    aff_ls: fn(&'a LinearProgram, &SolverOptions, &SolverState, &Step) -> (E, E),
    cc_ls: fn(&'a LinearProgram, &SolverOptions, &SolverState, &Step) -> (E, E),

    _solver: PhantomData<LinSolve>,
}

impl<'a, LinSolve: LinearSolver, Sys: AugmentedSystem<'a, LinSolve>, MU: MuUpdate<'a>>
    MehrotraPredictorCorrector<'a, LinSolve, Sys, MU>
{
    const DEFAULT_MAX_ITER: usize = 100;

    fn get_max_iter(&self) -> usize {
        if self.options.max_iterations < 1 {
            Self::DEFAULT_MAX_ITER
        } else {
            self.options.max_iterations as usize
        }
    }

    fn initialize(&mut self, _state: &mut SolverState) {
        // TODO: Initialization code here
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<(), Problem> {
        state.sigma = Some(E::from(0.));
        state.mu = Some(self.mu_updater.get(state));
        state.safety_factor = Some(E::from(1.));

        let mut rhs = self.lp.compute_residual(state);

        // Affine Step
        let aff_step = self.system.solve(state, &rhs)?;
        let (alpha_aff_primal, alpha_aff_dual) =
            (self.aff_ls)(self.lp, &self.options.root, state, &aff_step);

        // Center-Corrector Step
        let mut state_aff = state.clone();
        state_aff.x += alpha_aff_primal * &aff_step.dx;
        state_aff.y += alpha_aff_dual * &aff_step.dy;
        state_aff.z_l += alpha_aff_dual * &aff_step.dz_l;
        state_aff.z_u += alpha_aff_dual * &aff_step.dz_u;

        state.sigma = Some(pow(
            self.mu_updater.get(&state_aff) / state.mu.unwrap_or(E::from(1.)),
            3,
        ));
        state.safety_factor = Some(E::from(0.99)); // Reduce step length to maintain stability

        rhs.cs_lower -= cwise_multiply_finite(aff_step.dz_l.as_ref(), aff_step.dx.as_ref());
        rhs.cs_upper -= cwise_multiply_finite(aff_step.dz_u.as_ref(), aff_step.dx.as_ref());

        let corr_step = self.system.solve(state, &rhs)?;
        let (alpha_corr_primal, alpha_corr_dual) =
            (self.cc_ls)(self.lp, &self.options.root, state, &corr_step);

        // Update the state with the corrector step and step lengths
        state.x += alpha_corr_primal * &corr_step.dx;
        state.y += alpha_corr_dual * &corr_step.dy;
        state.z_l += alpha_corr_dual * &corr_step.dz_l;
        state.z_u += alpha_corr_dual * &corr_step.dz_u;
        state.alpha_primal = alpha_corr_primal;
        state.alpha_dual = alpha_corr_dual;

        state.residual = self.lp.compute_residual(state);
        state.status = Status::InProgress;

        Ok(())
    }
}

impl<'a, LinSolve: LinearSolver, Sys: AugmentedSystem<'a, LinSolve>, MU: MuUpdate<'a>> LPSolver<'a>
    for MehrotraPredictorCorrector<'a, LinSolve, Sys, MU>
{
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self {
        Self {
            lp,
            system: Sys::new(lp),
            mu_updater: MU::new(lp, options),

            aff_ls: line_search::compute_max_step_length,
            cc_ls: line_search::compute_max_step_length,

            options: options.into(),

            _solver: PhantomData,
        }
    }
}

impl<'a, LinSolve: LinearSolver, Sys: AugmentedSystem<'a, LinSolve>, MU: MuUpdate<'a>> Solver
    for MehrotraPredictorCorrector<'a, LinSolve, Sys, MU>
{
    /// Run the solver until convergence or maximum iterations.
    fn solve(
        &mut self,
        state: &mut SolverState,
        properties: &mut SolverHooks,
    ) -> Result<Status, Problem> {
        self.initialize(state);
        state.nit = 0;
        state.set_status(Status::InProgress);
        properties.callback.init(state);

        let max_iter = self.get_max_iter();
        for iter in 0..max_iter {
            state.nit = iter;
            self.iterate(state)?;

            let status = state.get_status();
            if status != Status::InProgress {
                println!(
                    "Converged in {} iterations with status: {:?}",
                    iter + 1,
                    status
                );
                return Ok(status);
            }

            properties.callback.call(state);
            if let Some(terminator_status) = properties.terminator.terminate(state) {
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
