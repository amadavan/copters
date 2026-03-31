use faer::{unzip, zip};
use macros::{explicit_options, use_option};

use crate::{
    E, SolverOptions,
    qp::{QuadraticProgram, pc::Workspace},
    state::{Delta, View},
};

pub trait MuUpdate {
    fn new(qp: &QuadraticProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;
    fn get(
        &mut self,
        view: &View<QuadraticProgram, Workspace>,
        alpha_primal: E,
        alpha_dual: E,
        delta: &Delta,
    ) -> E;
}

#[explicit_options(name = SolverOptions)]
#[use_option(name = "mu_fixed", type_ = E, default = "1.", description = "Constant value for the barrier parameter mu")]
pub struct ConstantMuUpdate {}

impl MuUpdate for ConstantMuUpdate {
    fn new(_qp: &QuadraticProgram, options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn get(
        &mut self,
        _view: &View<QuadraticProgram, Workspace>,
        alpha_primal: E,
        alpha_dual: E,
        delta: &Delta,
    ) -> E {
        self.options.mu_fixed
    }
}

pub struct MehrotraMuUpdate {}

impl MuUpdate for MehrotraMuUpdate {
    fn new(_qp: &QuadraticProgram, _options: &SolverOptions) -> Self {
        Self {}
    }

    fn get(
        &mut self,
        view: &View<QuadraticProgram, Workspace>,
        alpha_primal: E,
        alpha_dual: E,
        delta: &Delta,
    ) -> E {
        let View {
            program: qp,
            state,
            work,
        } = view;

        let x = state.variables().x();
        let z_l = state.variables().z_l();
        let z_u = state.variables().z_u();
        let l = &qp.l;
        let u = &qp.u;
        let dx = delta.dx();
        let dz_l = delta.dz_l();
        let dz_u = delta.dz_u();

        let mu = zip!(x, l, z_l, dx, dz_l)
            .map(|unzip!(x_i, l_i, z_l_i, dx_i, dz_l_i)| {
                if l_i.is_finite() && z_l_i.is_finite() {
                    (x_i + alpha_primal * dx_i - l_i) * (z_l_i + alpha_dual * dz_l_i)
                } else {
                    E::from(0.)
                }
            })
            .sum()
            + zip!(x, u, z_u, dx, dz_u)
                .map(|unzip!(x_i, u_i, z_u_i, dx_i, dz_u_i)| {
                    if u_i.is_finite() && z_u_i.is_finite() {
                        (x_i + alpha_primal * dx_i - u_i) * (z_u_i + alpha_dual * dz_u_i)
                    } else {
                        E::from(0.)
                    }
                })
                .sum();
        let mu_step = mu / E::from(x.nrows() as f64);
        mu_step * mu_step * mu_step / (work.mu + work.mu)
    }
}
