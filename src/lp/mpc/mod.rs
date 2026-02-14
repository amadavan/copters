use std::marker::PhantomData;

use faer::{Col, traits::num_traits::pow};
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, Properties, Solver, SolverOptions, SolverState, Status,
    linalg::{solver::LinearSolver, vector_ops::cwise_multiply_finite},
    lp::{
        LinearProgram, LinearProgramSolver,
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

/// KKT residuals for the current iterate.
///
/// ```text
/// dual_feasibility      = c - A^T y - z_l - z_u
/// primal_feasibility    = b - A x
/// cs_lower              = Z_l (x - l)
/// cs_upper              = Z_u (x - u)
/// ```
pub struct Residual {
    dual_feasibility: Col<E>,
    primal_feasibility: Col<E>,
    cs_lower: Col<E>,
    cs_upper: Col<E>,
}

impl Residual {
    pub fn get_dual_feasibility(&self) -> &Col<E> {
        &self.dual_feasibility
    }

    pub fn get_primal_feasibility(&self) -> &Col<E> {
        &self.primal_feasibility
    }

    pub fn get_complementarity_lower(&self) -> &Col<E> {
        &self.cs_lower
    }

    pub fn get_complementarity_upper(&self) -> &Col<E> {
        &self.cs_upper
    }
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
#[use_option(name = "MaxIterations", type_=usize, default="0", description="Maximum number of iterations (0 uses solver defaults).")]
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

    /// Computes the KKT residuals for the current primal-dual iterate.
    fn compute_residual(&self, state: &SolverState) -> Residual {
        // Compute the residuals based on the current state
        Residual {
            // Dual feasibility: c - A^T y - z_l - z_u
            dual_feasibility: &self.lp.c
                - self.lp.A.transpose() * &state.y
                - &state.z_l
                - &state.z_u,
            // Primal feasibility: b - A x
            primal_feasibility: &self.lp.b - &self.lp.A * &state.x,
            // Complimentary slackness
            cs_lower: -cwise_multiply_finite(state.z_l.as_ref(), (&state.x - &self.lp.l).as_ref()), // Placeholder
            cs_upper: -cwise_multiply_finite(state.z_u.as_ref(), (&state.x - &self.lp.u).as_ref()), // Placeholder
        }
    }

    fn get_max_iter(&self) -> usize {
        if self.options.MaxIterations < 1 {
            Self::DEFAULT_MAX_ITER
        } else {
            self.options.MaxIterations as usize
        }
    }

    fn initialize(&mut self, _state: &mut SolverState) {
        // TODO: Initialization code here
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<(), Problem> {
        // Iteration step code here

        state.set_sigma_mu(E::from(0.), self.mu_updater.get(state));
        state.set_safety_factor(E::from(1.));

        let mut residual = self.compute_residual(state);

        // Affine Step
        let aff_step = self.system.solve(state, &residual)?;
        let (alpha_aff_primal, alpha_aff_dual) =
            (self.aff_ls)(self.lp, &self.options.root, state, &aff_step);

        // Center-Corrector Step
        let mut state_aff = state.clone();
        state_aff.x += alpha_aff_primal * &aff_step.dx;
        state_aff.y += alpha_aff_dual * &aff_step.dy;
        state_aff.z_l += alpha_aff_dual * &aff_step.dz_l;
        state_aff.z_u += alpha_aff_dual * &aff_step.dz_u;

        state.set_sigma_mu(pow(self.mu_updater.get(&state_aff) / state.mu, 3), state.mu);
        state.set_safety_factor(E::from(0.99));

        residual.cs_lower -= cwise_multiply_finite(aff_step.dz_l.as_ref(), aff_step.dx.as_ref());
        residual.cs_upper -= cwise_multiply_finite(aff_step.dz_u.as_ref(), aff_step.dx.as_ref());

        let corr_step = self.system.solve(state, &residual)?;
        let (alpha_corr_primal, alpha_corr_dual) =
            (self.cc_ls)(self.lp, &self.options.root, state, &corr_step);

        // Update the state with the corrector step and step lengths
        state.x += alpha_corr_primal * &corr_step.dx;
        state.y += alpha_corr_dual * &corr_step.dy;
        state.z_l += alpha_corr_dual * &corr_step.dz_l;
        state.z_u += alpha_corr_dual * &corr_step.dz_u;
        state.alpha_primal = alpha_corr_primal;
        state.alpha_dual = alpha_corr_dual;

        let residual = self.compute_residual(state);
        state.primal_infeasibility = residual.get_primal_feasibility().norm_l2();
        state.dual_infeasibility = residual.get_dual_feasibility().norm_l2();
        state.complimentary_slack_lower = residual.get_complementarity_lower().norm_l2();
        state.complimentary_slack_upper = residual.get_complementarity_upper().norm_l2();

        state.status = Status::InProgress;

        Ok(())
    }
}

impl<'a, LinSolve: LinearSolver, Sys: AugmentedSystem<'a, LinSolve>, MU: MuUpdate<'a>>
    LinearProgramSolver<'a> for MehrotraPredictorCorrector<'a, LinSolve, Sys, MU>
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
        properties: &mut Properties,
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
