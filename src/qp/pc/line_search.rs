use faer::{unzip, zip};

use crate::{
    E, SolverOptions,
    qp::{QuadraticProgram, pc::Workspace},
    state::{Delta, View},
};

pub trait LineSearch {
    fn new(qp: &QuadraticProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Computes the maximum primal and dual step lengths that maintain feasibility.
    fn compute_step_length(&self, view: &View<QuadraticProgram, Workspace>, step: &Delta)
    -> (E, E);
}

pub struct PrimalDualFeasible {}

impl LineSearch for PrimalDualFeasible {
    fn new(_qp: &QuadraticProgram, _options: &SolverOptions) -> Self {
        Self {}
    }

    fn compute_step_length(
        &self,
        view: &View<QuadraticProgram, Workspace>,
        step: &Delta,
    ) -> (E, E) {
        let View {
            program: qp,
            state,
            work,
        } = view;

        // Identify the primal step size
        let alpha_primal = zip!(&qp.l, &qp.u, &state.vars.x(), &step.dx())
            .map(|unzip!(lb, ub, x, dx)| {
                if *dx < E::from(0.) {
                    if lb.is_finite() {
                        (lb - x) / dx
                    } else {
                        E::INFINITY
                    }
                } else if *dx > E::from(0.) {
                    if ub.is_finite() {
                        (ub - x) / dx
                    } else {
                        E::INFINITY
                    }
                } else {
                    E::INFINITY
                }
            })
            .min()
            .unwrap_or(E::from(1.));

        let alpha_dual = zip!(
            &state.vars.z_l(),
            &state.vars.z_u(),
            &step.dz_l(),
            &step.dz_u()
        )
        .map(|unzip!(zl, zu, dzl, dzu)| {
            let alpha_zl = if *dzl < E::from(0.) {
                if zl.is_finite() {
                    -zl / dzl
                } else {
                    E::INFINITY
                }
            } else {
                E::INFINITY
            };

            let alpha_zu = if *dzu > E::from(0.) {
                if zu.is_finite() {
                    -zu / dzu
                } else {
                    E::INFINITY
                }
            } else {
                E::INFINITY
            };

            E::min(alpha_zl, alpha_zu)
        })
        .min()
        .unwrap_or(E::from(1.));

        (
            E::min(E::from(1.), work.safety_factor * alpha_primal),
            E::min(E::from(1.), work.safety_factor * alpha_dual),
        )
    }
}
