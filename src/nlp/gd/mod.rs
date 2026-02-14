pub mod stepsize;

use faer::{unzip, zip};
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, I, Properties, Solver, SolverOptions, SolverState, Status,
    nlp::{NonlinearProgram, gd::stepsize::StepSize},
};

/// Projected gradient descent solver for nonlinear programs.
///
/// Solves problems of the form:
/// ```text
///   min  f(x)
///   s.t. g(x) = 0
///        l <= x <= u
/// ```
///
/// Uses a primal-dual gradient step: the primal variables `x` are updated along
/// the negative gradient of the Lagrangian, while the dual variables `y` are
/// updated via a gradient ascent step on the constraint violation. After each
/// step, `x` is projected onto the box `[l, u]`.
///
/// The step size strategy is configurable via the [`StepSize`] trait (e.g.
/// constant, linear decay, quadratic decay).
#[explicit_options(name = SolverOptions)]
#[use_option(name="learning_rate", type_=E, default="0.1", description="Learning rate for gradient descent.")]
#[use_option(name="max_iterations", type_=I, description="Maximum number of iterations for gradient descent.")]
pub struct GradientDescent<'a, SS: StepSize> {
    nlp: &'a NonlinearProgram,
    step: SS,
}

impl<'a, SS: StepSize> GradientDescent<'a, SS> {
    /// Creates a new gradient descent solver for the given nonlinear program.
    pub fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self {
        Self {
            nlp,
            step: SS::new(options),
            options: options.into(),
        }
    }

    /// Performs a single projected primal-dual gradient descent iteration.
    ///
    /// Updates `state.x` and `state.y` by taking a gradient step on the
    /// Lagrangian, projects `x` onto the bound constraints, and computes
    /// primal/dual infeasibility measures.
    fn iterate(&mut self, state: &mut SolverState) -> Result<Status, Problem> {
        let df = self.nlp.df(&state.x);
        let g = self.nlp.g(&state.x);
        let dg = self.nlp.dg(&state.x);

        let step_size = self.step.compute(state);
        state.x -= step_size * (&df + &dg.transpose() * &state.y); // Simple gradient step on the objective
        state.y += step_size * &g; // Simple gradient step on the constraints

        // Ensure feasibility of the primal variables
        if let Some(l) = self.nlp.l() {
            zip!(&mut state.x, l).for_each(|unzip!(x_i, l_i)| {
                if *x_i < *l_i {
                    *x_i = *l_i;
                }
            });
        }
        if let Some(u) = self.nlp.u() {
            zip!(&mut state.x, u).for_each(|unzip!(x_i, u_i)| {
                if *x_i > *u_i {
                    *x_i = *u_i;
                }
            });
        }

        // Update the state
        state.primal_infeasibility = g
            .iter()
            .filter(|&&val| val > 0.0)
            .fold(E::from(0.), |a, b| a + b);
        state.dual_infeasibility = df.iter().fold(E::from(0.), |a, b| a + b.abs());
        state.complimentary_slack_lower = 0.;
        state.complimentary_slack_upper = 0.;

        state.alpha_primal = step_size;
        state.alpha_dual = step_size;

        Ok(Status::InProgress) // Placeholder for actual status based on convergence criteria
    }
}

impl<'a, SS: StepSize> Solver for GradientDescent<'a, SS> {
    fn solve(
        &mut self,
        state: &mut SolverState,
        properties: &mut Properties,
    ) -> Result<Status, Problem> {
        let max_iterations = if self.options.max_iterations > 0 {
            self.options.max_iterations as I
        } else {
            1e6 as I // Default to a large number if not set
        };

        for iter in 0..max_iterations {
            state.nit = iter;

            let status = self.iterate(state)?;
            if status != Status::InProgress {
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
        Ok(Status::IterationLimit) // If max_iterations reached without convergence
    }
}

#[cfg(test)]
mod tests {
    use faer::sparse::{SparseColMat, Triplet};

    use crate::{
        callback::{Callback, ConvergenceOutput},
        nlp::gd::stepsize::ConstantStepSize,
        terminators::{MultiTerminator, SlowProgressTerminator, Terminator},
    };

    use super::*;

    #[test]
    fn test_gradient_descent() {
        let simple_nlp = NonlinearProgram::new(
            2,
            1,
            |x| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2), // Objective: minimize distance to (1, 2)
            |x| vec![x[0] + x[1] - 3.0].into_iter().collect(), // Constraint: x[0] + x[1] = 3
            |x| {
                vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]
                    .into_iter()
                    .collect()
            }, // Gradient of objective
            |_x| {
                let triplets = [Triplet::new(0, 0, 1.), Triplet::new(0, 1, 1.)];
                SparseColMat::<I, E>::try_new_from_triplets(1, 2, &triplets) // Jacobian of constraint
                    .unwrap()
            },
            None,
            None,
            None,
        );

        let mut state = SolverState {
            x: vec![0.0, 0.0].into_iter().collect(),
            y: vec![1.0].into_iter().collect(),
            nit: 0,
            status: Status::InProgress,
            alpha_primal: 0.0,
            alpha_dual: 0.0,

            primal_infeasibility: 0.0,
            dual_infeasibility: 0.0,
            complimentary_slack_lower: 0.0,
            complimentary_slack_upper: 0.0,

            mu: 0.0,
            sigma: 0.0,
            safety_factor: 0.0,
            z_l: vec![0.0].into_iter().collect(),
            z_u: vec![0.0].into_iter().collect(),
        };

        let options = SolverOptions::new();
        let mut properties = Properties {
            callback: Box::new(ConvergenceOutput::new(&options)),
            terminator: Box::new(SlowProgressTerminator::new(&options)),
        };

        let mut gd_solver = GradientDescent::<ConstantStepSize>::new(&simple_nlp, &options);
        let result = gd_solver.solve(&mut state, &mut properties).unwrap();
        assert_eq!(result, Status::Optimal);
        assert!((state.x[0] - 1.0).abs() < 1e-3);
        assert!((state.x[1] - 2.0).abs() < 1e-3);
    }
}
