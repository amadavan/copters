pub mod stepsize;

use faer::{Col, ColRef, sparse::SparseColMat, unzip, zip};
use macros::{explicit_options, use_option};
use problemo::Problem;

use crate::{
    E, I, IterativeSolver, OptimizationProgram, SolverOptions,
    nlp::{NLPSolver, NonlinearProgram, gd::stepsize::StepSize},
    state::{self, SolverState, Status, View},
};

#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct Workspace {
    _f: E,
    g: Col<E>,
    df: Col<E>,
    dg: SparseColMat<I, E>,
    _h: Option<SparseColMat<I, E>>,
    L: E,
    dL: Col<E>,
}

impl Workspace {
    fn new(n: usize, m: usize) -> Self {
        Self {
            _f: E::from(0.),
            g: Col::zeros(m),
            df: Col::zeros(n),
            dg: SparseColMat::try_new_from_nonnegative_triplets(n, m, &[]).unwrap(),
            _h: None,
            L: 0.,
            dL: Col::zeros(n),
        }
    }

    fn update<'a>(&mut self, nlp: &'a NonlinearProgram, x: &ColRef<E>, y: &ColRef<E>) {
        // self.f = nlp.f(x);
        self.g = nlp.g(x);
        self.df = nlp.df(x);
        self.dg = nlp.dg(x);
        self.L = nlp.f(x) + nlp.g(x).transpose() * y;
        self.dL = (self.df.as_ref() + self.dg.as_ref().transpose() * y).to_owned();
    }

    #[allow(non_snake_case)]
    fn compute_L(
        &self,
        nlp: &NonlinearProgram,
        state: &SolverState,
        alpha_primal: E,
        alpha_dual: E,
    ) -> E {
        let step_x = &state.vars.x() + alpha_primal * &state.delta.dx();
        let step_y = &state.vars.y() + alpha_dual * &state.delta.dy();
        nlp.f(&step_x.as_ref()) + nlp.g(&step_x.as_ref()).transpose() * &step_y.as_ref()
    }
}

impl state::Workspace for Workspace {
    fn new<'a, P: OptimizationProgram>(_program: &'a P, state: &'a mut SolverState) -> Self {
        Self::new(state.variables().x().nrows(), state.variables().y().nrows())
    }
}

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
#[use_option(name="max_iterations", type_=I, default="0", description="Maximum number of iterations for gradient descent.")]
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
    fn iterate(&mut self, view: &mut View<NonlinearProgram, Workspace>) -> Result<Status, Problem> {
        // Update the workspace
        let View {
            program: _program,
            state,
            work,
        } = view; // Destructure to access program and state

        work.update(self.nlp, &state.vars.x(), &state.vars.y());

        state.delta.dx_mut().copy_from(-&work.dL); // Store the primal gradient in the delta for potential use in line search or diagnostics
        state.delta.dy_mut().copy_from(&work.g); // Store the constraint violation in the delta for potential use in line search or diagnostics
        let (primal_step, dual_step) = self.step.compute(&self.nlp, &state, work);

        state.vars.update(primal_step, dual_step, &state.delta);

        // Ensure feasibility of the primal variables
        if let Some(l) = self.nlp.l() {
            zip!(&mut state.vars.x_mut(), l).for_each(|unzip!(x_i, l_i)| {
                if *x_i < *l_i {
                    *x_i = *l_i;
                }
            });
        }
        if let Some(u) = self.nlp.u() {
            zip!(&mut state.vars.x_mut(), u).for_each(|unzip!(x_i, u_i)| {
                if *x_i > *u_i {
                    *x_i = *u_i;
                }
            });
        }

        // Update the state
        state.residuals.primal_mut().copy_from(&work.g); // Update primal residual with constraint violation
        state.residuals.dual_mut().copy_from(&work.dL); // Update dual residual with gradient of the Lagrangian
        state.alpha_primal = primal_step;
        state.alpha_dual = dual_step;

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
    type Program = NonlinearProgram;
    type Workspace = Workspace;

    fn get_max_iterations(&self) -> usize {
        if self.options.max_iterations as usize > 0 {
            self.options.max_iterations as usize
        } else {
            1000
        }
    }

    fn iterate(&mut self, view: &mut View<NonlinearProgram, Workspace>) -> Result<Status, Problem> {
        self.iterate(view)
    }
}

#[cfg(test)]
mod tests {
    use faer::sparse::{SparseColMat, Triplet};
    use rstest::{fixture, rstest};

    use crate::terminators::ConvergenceTerminator;
    use crate::{SolverHooks, callback::ConvergenceOutput, terminators::SlowProgressTerminator};

    use super::stepsize::*;
    use super::*;

    #[fixture]
    fn build_simple_nlp() -> (NonlinearProgram, SolverState) {
        let nlp = NonlinearProgram::new(
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
            None::<fn(&ColRef<E>, &ColRef<E>) -> SparseColMat<I, E>>,
            None,
            None,
        );

        let mut state = SolverState::new(2, 1);
        state
            .variables_mut()
            .x_mut()
            .copy_from(&vec![0.0, 0.0].into_iter().collect::<Col<E>>());
        state
            .variables_mut()
            .y_mut()
            .copy_from(&vec![1.0].into_iter().collect::<Col<E>>());
        state
            .variables_mut()
            .z_l_mut()
            .copy_from(&vec![0.0, 0.0].into_iter().collect::<Col<E>>());
        state
            .variables_mut()
            .z_u_mut()
            .copy_from(&vec![0.0, 0.0].into_iter().collect::<Col<E>>());

        (nlp, state)
    }

    #[rstest]
    fn gd_constant() {
        let (simple_nlp, mut state) = build_simple_nlp();

        let options = SolverOptions::new();
        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(SlowProgressTerminator::new(&options)),
        };

        let mut gd_solver = GradientDescent::<ConstantStepSize>::new(&simple_nlp, &options);
        let result = gd_solver
            .optimize(&simple_nlp, &mut state, &mut properties)
            .unwrap();
        assert_eq!(result, Status::Optimal);
        assert!((state.variables().x()[0] - 1.0).abs() < 1e-3);
        assert!((state.variables().x()[1] - 2.0).abs() < 1e-3);
    }

    #[rstest]
    fn gd_linear_decay() {
        let (simple_nlp, mut state) = build_simple_nlp();

        let mut options = SolverOptions::new();
        let _ = options.set_option("learning_rate", 10.0);
        let _ = options.set_option("max_iterations", 100);
        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(SlowProgressTerminator::new(&options)),
        };

        let mut gd_solver = GradientDescent::<LinearDecayStepSize>::new(&simple_nlp, &options);
        let result = gd_solver
            .optimize(&simple_nlp, &mut state, &mut properties)
            .unwrap();
        assert_eq!(result, Status::Optimal);
        assert!((state.variables().x()[0] - 1.0).abs() < 1e-3);
        assert!((state.variables().x()[1] - 2.0).abs() < 1e-3);
    }

    #[rstest]
    fn gd_barzilai_borwein(
        #[values(
            BarzilaiBorweinVariant::ShortStep,
            BarzilaiBorweinVariant::LongStep,
            BarzilaiBorweinVariant::Hybrid
        )]
        variant: BarzilaiBorweinVariant,
    ) {
        let (simple_nlp, mut state) = build_simple_nlp();

        let mut options = SolverOptions::new();
        options
            .set_option("barzilai_borwein_variant", variant)
            .unwrap();
        options.set_option("max_iterations", 1000 as I).unwrap();
        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut gd_solver = GradientDescent::<BarzilaiBorwein>::new(&simple_nlp, &options);
        let result = gd_solver
            .optimize(&simple_nlp, &mut state, &mut properties)
            .unwrap();
        assert_eq!(result, Status::Optimal);
        assert!((state.variables().x()[0] - 1.0).abs() < 1e-3);
        assert!((state.variables().x()[1] - 2.0).abs() < 1e-3);
    }

    #[rstest]
    fn gd_armijo() {
        let (simple_nlp, mut state) = build_simple_nlp();

        let mut options = SolverOptions::new();
        let _ = options.set_option("learning_rate", 10.0);
        let _ = options.set_option("max_iterations", 100 as I);
        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut gd_solver = GradientDescent::<ArmijoRule>::new(&simple_nlp, &options);
        let result = gd_solver
            .optimize(&simple_nlp, &mut state, &mut properties)
            .unwrap();
        assert_eq!(result, Status::Optimal);
        assert!((state.variables().x()[0] - 1.0).abs() < 1e-3);
        assert!((state.variables().x()[1] - 2.0).abs() < 1e-3);
    }
}
