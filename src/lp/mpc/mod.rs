use std::marker::PhantomData;

use faer::{Col, traits::num_traits::pow};
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, IterativeSolver, SolverOptions, SolverState, Status,
    callback::{Callback, Callbacks},
    linalg::{solver::LinearSolver, vector_ops::cwise_multiply_finite},
    lp::{
        LinearProgram, LinearProgramSolver,
        mpc::{augmented_system::AugmentedSystem, line_search::LineSearch, mu_update::MuUpdate},
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
    Solver: LinearSolver,
    System: AugmentedSystem<'a, Solver>,
    MU: MuUpdate<'a>,
    LS: LineSearch<'a>,
> {
    lp: &'a LinearProgram,

    system: System,
    mu_updater: MU,
    line_search: LS,

    _solver: PhantomData<Solver>,
}

impl<
    'a,
    Solver: LinearSolver,
    System: AugmentedSystem<'a, Solver>,
    MU: MuUpdate<'a>,
    LS: LineSearch<'a>,
> MehrotraPredictorCorrector<'a, Solver, System, MU, LS>
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
            cs_lower: cwise_multiply_finite(state.z_l.as_ref(), (&state.x - &self.lp.l).as_ref()), // Placeholder
            cs_upper: cwise_multiply_finite(state.z_u.as_ref(), (&state.x - &self.lp.u).as_ref()), // Placeholder
        }
    }
}

impl<
    'a,
    Solver: LinearSolver,
    System: AugmentedSystem<'a, Solver>,
    MU: MuUpdate<'a>,
    LS: LineSearch<'a>,
> LinearProgramSolver<'a> for MehrotraPredictorCorrector<'a, Solver, System, MU, LS>
{
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self {
        Self {
            lp,
            system: System::new(lp),
            mu_updater: MU::new(lp, options),
            line_search: LS::new(lp, options),
            options: options.into(),

            _solver: PhantomData,
        }
    }
}

impl<
    'a,
    Solver: LinearSolver,
    System: AugmentedSystem<'a, Solver>,
    MU: MuUpdate<'a>,
    LS: LineSearch<'a>,
> IterativeSolver for MehrotraPredictorCorrector<'a, Solver, System, MU, LS>
{
    fn get_max_iter(&self) -> usize {
        if self.options.MaxIterations < 1 {
            Self::DEFAULT_MAX_ITER
        } else {
            self.options.MaxIterations as usize
        }
    }

    fn initialize(&mut self, state: &mut SolverState) {
        // Initialization code here
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<(), Problem> {
        // Iteration step code here

        state.set_sigma_mu(Some(1.), Some(self.mu_updater.get(state)));

        let mut residual = self.compute_residual(state);

        // Affine Step
        let aff_step = self.system.solve(state, &residual)?;
        let alpha_aff_primal = self.line_search.get_primal_step_length(state, &aff_step);
        let alpha_aff_dual = self.line_search.get_dual_step_length(state, &aff_step);

        // Center-Corrector Step
        let mut state_aff = state.clone();
        state_aff.x += alpha_aff_primal * &aff_step.dx;
        state_aff.y += alpha_aff_dual * &aff_step.dy;
        state_aff.z_l += alpha_aff_dual * &aff_step.dz_l;
        state_aff.z_u += alpha_aff_dual * &aff_step.dz_u;

        state.set_sigma_mu(
            Some(pow(self.mu_updater.get(&state_aff) / state.mu.unwrap(), 3)),
            state.mu,
        );

        residual.cs_lower -= cwise_multiply_finite(aff_step.dz_l.as_ref(), aff_step.dx.as_ref());
        residual.cs_upper -= cwise_multiply_finite(aff_step.dz_u.as_ref(), aff_step.dx.as_ref());

        let corr_step = self.system.solve(state, &residual)?;
        let alpha_corr_primal = self.line_search.get_primal_step_length(state, &corr_step);
        let alpha_corr_dual = self.line_search.get_dual_step_length(state, &corr_step);

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

        state.status = Status::InProgress;

        Ok(())
    }
}
