use std::ops::SubAssign;

use macros::explicit_options;
use problemo::Problem;

use crate::{
    E, I, IterativeSolver, OptimizationProgram, Solver, SolverHooks, SolverOptions,
    linalg::vector_ops::cwise_multiply,
    qp::{
        QuadraticProgram,
        pc::{augmented_system::AugmentedSystem, line_search::LineSearch, mu_update::MuUpdate},
    },
    state::{self, Delta, SolverState, Status, View},
};

pub mod augmented_system;
pub mod line_search;
pub mod mu_update;

#[derive(Debug)]
pub struct Workspace {
    pub mu: E,
    pub sigma: E,
    pub safety_factor: E,
    pub rhs: Delta,
}

impl Workspace {
    fn new(n: I, m: I) -> Self {
        Self {
            mu: 1.0,
            sigma: 0.1,
            safety_factor: 1.0,
            rhs: Delta::new(n, m), // Placeholder, will be initialized properly in the solver
        }
    }
}

impl state::Workspace for Workspace {
    fn new<'a, P: OptimizationProgram>(
        _program: &'a P,
        state: &'a mut crate::state::SolverState,
    ) -> Self {
        let n = state.variables().x().nrows();
        let m = state.variables().y().nrows();
        Self::new(n, m)
    }
}

#[explicit_options(name = SolverOptions)]
pub struct PredictorCorrector<AS: AugmentedSystem, LS: LineSearch, MU: MuUpdate> {
    aug_sys: AS,
    line_search: LS,
    mu_update: MU,
}

impl<AS: AugmentedSystem, LS: LineSearch, MU: MuUpdate> PredictorCorrector<AS, LS, MU> {
    pub fn new(qp: &QuadraticProgram, options: &SolverOptions) -> Self {
        Self {
            aug_sys: AS::new(&options),
            line_search: LS::new(qp, &options),
            mu_update: MU::new(qp, &options),

            options: options.into(),
        }
    }
}

impl<AS: AugmentedSystem, LS: LineSearch, MU: MuUpdate> IterativeSolver
    for PredictorCorrector<AS, LS, MU>
{
    type Workspace = Workspace;

    fn get_max_iterations(&self) -> usize {
        100
    }

    fn initialize(&mut self, view: &mut crate::state::View<Self::Program, Self::Workspace>) {
        // Initialize the workspace with default values
        self.aug_sys.analyze_sys(view);
    }

    fn iterate(
        &mut self,
        view: &mut state::View<Self::Program, Self::Workspace>,
    ) -> Result<Status, Problem> {
        // Compute the RHS (assign as Delta for convenience)
        {
            let View {
                program: qp,
                state,
                work,
            } = view;

            work.rhs
                .data
                .copy_from_slice(state.residuals().data.as_slice());
        }

        // Affine step
        self.aug_sys.factorize_sys(view);
        view.work_mut().sigma = 0.;
        let delta_aff = self.aug_sys.solve(view);

        view.work_mut().safety_factor = 1.0;
        let (alpha_aff_primal, alpha_aff_dual) =
            self.line_search.compute_step_length(view, &delta_aff);

        // Centering step
        view.work_mut().sigma = 1.0;
        view.work_mut().mu = self
            .mu_update
            .get(view, alpha_aff_primal, alpha_aff_dual, &delta_aff);

        // Corrector step
        view.work_mut()
            .rhs
            .dz_l_mut()
            .sub_assign(cwise_multiply(delta_aff.dz_l(), delta_aff.dx()));
        view.work_mut()
            .rhs
            .dz_u_mut()
            .sub_assign(cwise_multiply(delta_aff.dz_u(), delta_aff.dx()));
        let delta_corr = self.aug_sys.solve(view);

        view.work_mut().safety_factor = 0.9995;
        let (alpha_corr_primal, alpha_corr_dual) =
            self.line_search.compute_step_length(view, &delta_corr);

        // Update the state with the corrector step and step lengths
        view.state_mut()
            .update(alpha_corr_primal, alpha_corr_dual, &delta_corr);

        Ok(Status::InProgress)
    }
}

impl<AS: AugmentedSystem, LS: LineSearch, MU: MuUpdate> Solver for PredictorCorrector<AS, LS, MU> {
    type Program = QuadraticProgram;

    fn solve(
        &mut self,
        program: &Self::Program,
        state: &mut SolverState,
        hooks: &mut SolverHooks,
    ) -> Result<Status, Problem> {
        self.solve_impl(program, state, hooks)
    }
}
