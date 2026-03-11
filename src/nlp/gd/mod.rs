pub mod stepsize;

use faer::{unzip, zip};
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, I, IterativeSolver, OptimizationProgram, SolverOptions, SolverState, Status, ipm,
    nlp::{NLPSolver, NonlinearProgram, gd::stepsize::StepSize},
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
    /// Performs a single projected primal-dual gradient descent iteration.
    ///
    /// Updates `state.x` and `state.y` by taking a gradient step on the
    /// Lagrangian, projects `x` onto the bound constraints, and computes
    /// primal/dual infeasibility measures.
    fn iterate(&mut self, state: &mut SolverState) -> Result<Status, Problem> {
        state.df = Some(self.nlp.df(&state.x));
        state.g = Some(self.nlp.g(&state.x));
        state.dg = Some(self.nlp.dg(&state.x));

        state.dL =
            Some(state.df.as_ref().unwrap() + state.dg.as_ref().unwrap().transpose() * &state.y); // Gradient of the Lagrangian w.r.t. x

        let step_size = self.step.compute(state);
        state.x -= step_size * state.dL.as_ref().unwrap(); // Simple gradient step on the objective
        state.y += step_size * state.g.as_ref().unwrap(); // Simple gradient step on the constraints

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
        self.nlp.update_residual(state);

        state.alpha_primal = step_size;
        state.alpha_dual = step_size;

        Ok(Status::InProgress) // Placeholder for actual status based on convergence criteria
    }
}

impl<'a, SS: StepSize> NLPSolver<'a> for GradientDescent<'a, SS> {
    /// Creates a new gradient descent solver for the given nonlinear program.
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self {
        Self {
            nlp,
            step: SS::new(options),
            options: options.into(),
        }
    }
}

impl<'a, SS: StepSize> IterativeSolver for GradientDescent<'a, SS> {
    fn get_max_iterations(&self) -> usize {
        if self.options.max_iterations as usize > 0 {
            self.options.max_iterations as usize
        } else {
            ipm::DEFAULT_MAX_ITERATIONS
        }
    }

    fn get_program(&self) -> &dyn OptimizationProgram {
        self.nlp
    }

    fn iterate(&mut self, state: &mut SolverState) -> Result<Status, Problem> {
        self.iterate(state)
    }
}

#[cfg(test)]
mod tests {
    use faer::sparse::{SparseColMat, Triplet};

    use crate::{
        callback::ConvergenceOutput, nlp::gd::stepsize::ConstantStepSize,
        terminators::SlowProgressTerminator,
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

        let mut state = SolverState::new(
            vec![0.0, 0.0].into_iter().collect(),
            vec![1.0].into_iter().collect(),
            vec![0.0, 0.0].into_iter().collect(),
            vec![0.0, 0.0].into_iter().collect(),
        );

        let options = SolverOptions::new();
        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(SlowProgressTerminator::new(&options)),
        };

        let mut gd_solver = GradientDescent::<ConstantStepSize>::new(&simple_nlp, &options);
        let result = gd_solver.solve(&mut state, &mut properties).unwrap();
        assert_eq!(result, Status::Optimal);
        assert!((state.x[0] - 1.0).abs() < 1e-3);
        assert!((state.x[1] - 2.0).abs() < 1e-3);
    }
}
