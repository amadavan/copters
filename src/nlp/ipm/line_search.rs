use faer::{Col, unzip, zip};
use macros::explicit_options;

use crate::{
    E, SolverOptions, SolverState,
    nlp::{NonlinearProgram, ipm::Step},
};

pub trait LineSearch<'a> {
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Performs a line search to find an appropriate step size `alpha` for the given search direction.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state of the solver, containing the current iterate and other relevant information.
    /// * `properties` - The properties of the problem being solved, which may include parameters for the line search.
    /// * `search_direction` - The search direction along which to perform the line search.
    ///
    /// # Returns
    ///
    /// A step size `alpha` that satisfies the line search criteria.
    fn perform_line_search(&self, state: &SolverState, search_direction: &Step) -> E;
}

#[explicit_options(name = SolverOptions)]
pub struct BacktrackingLineSearch<'a> {
    // Add any necessary fields for the backtracking line search strategy
    nlp: &'a NonlinearProgram,
}

impl<'a> LineSearch<'a> for BacktrackingLineSearch<'a> {
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized,
    {
        // Initialize any necessary fields based on the solver options
        Self {
            nlp,
            options: options.into(),
        }
    }

    fn perform_line_search(&self, state: &SolverState, search_direction: &Step) -> E {
        // Implement the backtracking line search logic here
        // This typically involves starting with an initial step size and iteratively reducing it until a certain condition is met

        // Placeholder implementation
        1.0 // Return a dummy value for now
    }
}

#[explicit_options(name = SolverOptions)]
pub struct PDFeasibileLineSearch<'a> {
    // Add any necessary fields for the primal-dual feasible line search strategy
    nlp: &'a NonlinearProgram,
}

impl<'a> LineSearch<'a> for PDFeasibileLineSearch<'a> {
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self
    where
        Self: Sized,
    {
        // Initialize any necessary fields based on the solver options
        Self {
            nlp,
            options: options.into(),
        }
    }

    fn perform_line_search(&self, state: &SolverState, step: &Step) -> E {
        // Implement the primal-dual feasible line search logic here
        let zero = Col::<E>::zeros(state.x.nrows());
        let inf = E::INFINITY * Col::<E>::ones(state.x.nrows());
        let l = self.nlp.l.as_ref().unwrap_or(&zero);
        let u = self.nlp.u.as_ref().unwrap_or(&inf);

        let xl = &state.x - l;
        let xu = &state.x - u;

        let mut alpha_primal = E::from(1.);
        zip!(&xl, &xu, l, u, &step.dx).for_each(|unzip!(xl, xu, l, u, dx)| {
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

        let alpha_primal = E::min(E::from(1.), state.safety_factor.unwrap() * alpha_primal);
        let alpha_dual = E::min(E::from(1.), state.safety_factor.unwrap() * alpha_dual);

        E::min(alpha_primal, alpha_dual)
    }
}
