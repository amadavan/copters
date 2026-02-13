use faer::{unzip, zip};

use crate::{
    E, SolverOptions, SolverState,
    lp::{LinearProgram, mpc::Step},
};

pub fn compute_max_step_length<'a>(
    lp: &'a LinearProgram,
    _options: &SolverOptions,
    state: &SolverState,
    step: &Step,
) -> (E, E) {
    let xl = &state.x - &lp.l;
    let xu = &state.x - &lp.u;

    let mut alpha_primal = E::from(1.);
    zip!(&xl, &xu, &lp.l, &lp.u, &step.dx).for_each(|unzip!(xl, xu, l, u, dx)| {
        let dx_neg = if *dx < E::from(0.) { *dx } else { -*xl };
        let dx_pos = if *dx > E::from(0.) { *dx } else { -*xu };

        let alpha_lb = if l.is_finite() {
            -xl / dx_neg
        } else {
            E::INFINITY
        };
        let alpha_ub = if u.is_finite() {
            -xu / dx_pos
        } else {
            E::INFINITY
        };

        alpha_primal = E::min(alpha_primal, E::min(alpha_lb, alpha_ub));
    });

    let mut alpha_dual = E::from(1.);
    zip!(&state.z_u, &step.dz_u).for_each(|unzip!(z_ub, dz_ub)| {
        let dz_ub_pos = if *dz_ub > E::from(0.) { *dz_ub } else { -*z_ub };
        let alpha_ub = if *z_ub < E::from(0.) {
            -z_ub / dz_ub_pos
        } else {
            E::INFINITY
        };

        alpha_dual = E::min(alpha_dual, alpha_ub);
    });

    zip!(&state.z_l, &step.dz_l).for_each(|unzip!(z_lb, dz_lb)| {
        let dz_lb_neg = if *dz_lb < E::from(0.) { *dz_lb } else { -*z_lb };
        let alpha_lb = if *z_lb > E::from(0.) {
            -z_lb / dz_lb_neg
        } else {
            E::INFINITY
        };

        alpha_dual = E::min(alpha_dual, alpha_lb);
    });

    let alpha_primal = E::min(E::from(1.), state.safety_factor * alpha_primal);
    let alpha_dual = E::min(E::from(1.), state.safety_factor * alpha_dual);

    (alpha_primal, alpha_dual)
}
