use std::marker::PhantomData;

use faer::traits::num_traits::pow;
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, I, IterativeSolver, OptimizationProgram, SearchDirection, SolverOptions, SolverState,
    Status,
    ipm::{self, RHS},
    linalg::{solver::LinearSolver, vector_ops::cwise_multiply_finite},
    qp::{
        QPSolver, QuadraticProgram,
        mpc::{augmented_system::AugmentedSystem, mu_update::MuUpdate},
    },
};

pub mod augmented_system;
pub mod line_search;
pub mod mu_update;

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
#[use_option(name = "max_iterations", type_=I, description="Maximum number of iterations (0 uses solver defaults).")]
pub struct MehrotraPredictorCorrector<
    'a,
    LinSolve: LinearSolver,
    Sys: AugmentedSystem<'a, LinSolve>,
    MU: MuUpdate<'a>,
> {
    qp: &'a QuadraticProgram,

    system: Sys,
    mu_updater: MU,

    aff_ls: fn(&'a QuadraticProgram, &SolverOptions, &SolverState, &SearchDirection) -> (E, E),
    cc_ls: fn(&'a QuadraticProgram, &SolverOptions, &SolverState, &SearchDirection) -> (E, E),

    _solver: PhantomData<LinSolve>,
}

impl<'a, LinSolve: LinearSolver, Sys: AugmentedSystem<'a, LinSolve>, MU: MuUpdate<'a>>
    MehrotraPredictorCorrector<'a, LinSolve, Sys, MU>
{
    fn initialize(&mut self, _state: &mut SolverState) {
        // TODO: Initialization code here
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<(), Problem> {
        // Iteration step code here

        state.sigma = Some(E::from(0.));
        state.mu = Some(self.mu_updater.get(state));
        state.safety_factor = Some(E::from(1.));

        // Compute RHS from residual
        let mut rhs = RHS::from(&*state);

        // Affine Step
        let aff_step = self.system.solve(state, &rhs)?;
        let (alpha_aff_primal, alpha_aff_dual) =
            (self.aff_ls)(self.qp, &self.options.root, state, &aff_step);

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

        *rhs.r_l_mut() -=
            cwise_multiply_finite(aff_step.get_dz_l().as_ref(), aff_step.get_dx().as_ref());
        *rhs.r_u_mut() -=
            cwise_multiply_finite(aff_step.get_dz_u().as_ref(), aff_step.get_dx().as_ref());

        let corr_step = self.system.solve(state, &rhs)?;
        let (alpha_corr_primal, alpha_corr_dual) =
            (self.cc_ls)(self.qp, &self.options.root, state, &corr_step);

        // Update the state with the corrector step and step lengths
        state.x += alpha_corr_primal * &corr_step.dx;
        state.y += alpha_corr_dual * &corr_step.dy;
        state.z_l += alpha_corr_dual * &corr_step.dz_l;
        state.z_u += alpha_corr_dual * &corr_step.dz_u;
        state.alpha_primal = alpha_corr_primal;
        state.alpha_dual = alpha_corr_dual;

        self.qp.update_residual(state);
        state.status = Status::InProgress;

        Ok(())
    }
}

impl<'a, LinSolve: LinearSolver, Sys: AugmentedSystem<'a, LinSolve>, MU: MuUpdate<'a>> QPSolver<'a>
    for MehrotraPredictorCorrector<'a, LinSolve, Sys, MU>
{
    fn new(qp: &'a QuadraticProgram, options: &SolverOptions) -> Self {
        Self {
            qp,
            system: Sys::new(qp),
            mu_updater: MU::new(qp, options),

            aff_ls: line_search::compute_max_step_length,
            cc_ls: line_search::compute_max_step_length,

            options: options.into(),

            _solver: PhantomData,
        }
    }
}

impl<'a, LinSolve: LinearSolver, Sys: AugmentedSystem<'a, LinSolve>, MU: MuUpdate<'a>>
    IterativeSolver for MehrotraPredictorCorrector<'a, LinSolve, Sys, MU>
{
    fn get_max_iterations(&self) -> usize {
        if self.options.max_iterations as usize > 0 {
            self.options.max_iterations as usize
        } else {
            ipm::DEFAULT_MAX_ITERATIONS
        }
    }

    fn get_program(&self) -> &dyn OptimizationProgram {
        self.qp
    }

    fn initialize(&mut self, state: &mut SolverState) {
        self.initialize(state);
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<Status, Problem> {
        self.iterate(state)?;
        Ok(state.get_status())
    }
}
