use faer::{Col, unzip, zip};
use macros::{build_option_enum, explicit_options, use_option};

use crate::{
    E, SolverOptions, SolverState,
    lp::{LinearProgram, mpc::Step},
};

/// Strategy for computing primal and dual step lengths in a primal-dual
/// interior-point method.
///
/// Implementations determine the maximum step size along a given direction
/// that keeps the iterate strictly feasible.
pub trait LineSearch<'a> {
    /// Creates a new instance from the linear program and solver options.
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Returns the largest primal step length `alpha` in `(0, 1]` such that
    /// `l <= x + alpha * dx <= u` remains strictly feasible.
    fn get_primal_step_length(&self, state: &SolverState, step: &Step) -> E;

    /// Returns the largest dual step length `alpha` in `(0, 1]` such that
    /// `z_l + alpha * dz_l >= 0` and `z_u + alpha * dz_u <= 0`.
    fn get_dual_step_length(&self, state: &SolverState, step: &Step) -> E;
}

/// Line search strategy for primal-dual interior point methods in linear programming.
///
/// `LPLineSearch` implements the `LineSearch` trait and provides methods to compute
/// the maximum feasible step length for primal and dual variables, ensuring strict
/// feasibility by applying a safety factor. It operates on the current state and step
/// direction, and is used to maintain feasibility during optimization iterations.
///
/// # Options
/// - `safety_factor` (f64, default: 0.999): Multiplier to ensure steps remain strictly feasible.
///
/// # Usage
/// Use `LPLineSearch` when you need a robust line search for primal-dual IPM solvers.
/// Construct with options, then call `get_primal_step_length` and `get_dual_step_length` as needed.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "safety_factor", type_ = f64, default = "0.999", description = "Safety factor to ensure strict feasibility in line search")]
pub struct LPLineSearch<'a> {
    lp: &'a LinearProgram,
}

impl<'a> LineSearch<'a> for LPLineSearch<'a> {
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self {
        Self {
            lp,
            options: options.into(),
        }
    }

    fn get_primal_step_length(&self, state: &SolverState, step: &Step) -> E {
        let mut dx_min = Col::zeros(state.x.nrows());

        zip!(&state.x, &self.lp.l, &self.lp.u, &step.dx, &mut dx_min).for_each(
            |unzip!(x, l, u, dx, dx_min)| {
                let dx_neg = if *dx < E::from(0.) { *dx } else { -x };
                let dx_pos = if *dx > E::from(0.) { *dx } else { -x };

                let alpha_lb = if l.is_finite() {
                    -(x - l) / dx_neg
                } else {
                    E::INFINITY
                };
                let alpha_ub = if u.is_finite() {
                    -(x - u) / dx_pos
                } else {
                    E::INFINITY
                };

                *dx_min = E::min(alpha_lb, alpha_ub);
            },
        );

        let alpha_primal = E::min(
            E::from(1.),
            self.options.safety_factor * crate::linalg::vector_ops::col_min(dx_min.as_ref()),
        );

        alpha_primal
    }

    fn get_dual_step_length(&self, state: &SolverState, step: &Step) -> E {
        let mut ds_ub_pos = Col::zeros(state.x.nrows());
        let mut ds_lb_neg = Col::zeros(state.x.nrows());

        // Only get negatives elemented of ds_ub and ds_lb
        zip!(&state.z_u, &step.dz_u, &mut ds_ub_pos).for_each(|unzip!(s_ub, ds_ub, ds_pos)| {
            *ds_pos = if *ds_ub > E::from(0.) {
                -*ds_ub
            } else {
                E::from(*s_ub)
            }
        });
        let alpha_dual_ub = E::min(
            E::from(1.),
            self.options.safety_factor
                * crate::linalg::vector_ops::col_min(
                    crate::linalg::vector_ops::cwise_quotient(
                        state.z_u.as_ref(),
                        ds_ub_pos.as_ref(),
                    )
                    .as_ref(),
                ),
        );

        zip!(&state.z_l, &step.dz_l, &mut ds_lb_neg).for_each(|unzip!(s_lb, ds_lb, ds_neg)| {
            *ds_neg = if *ds_lb < E::from(0.) {
                -*ds_lb
            } else {
                E::from(*s_lb)
            }
        });
        let alpha_dual_lb = E::min(
            E::from(1.),
            self.options.safety_factor
                * crate::linalg::vector_ops::col_min(
                    crate::linalg::vector_ops::cwise_quotient(
                        state.z_l.as_ref(),
                        ds_lb_neg.as_ref(),
                    )
                    .as_ref(),
                ),
        );

        E::min(alpha_dual_ub, alpha_dual_lb)
    }
}

// build_option_enum!(
//     trait_ = for <'a> LineSearch<'a>,
//     name = "MPCLineSearch",
//     variants = (LPLineSearch,),
//     new_arguments = (&LinearProgram, &SolverOptions,),
//     doc_header = "Line search strategies for primal-dual interior point methods."
// );
