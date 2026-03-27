use macros::{explicit_options, use_option};

use faer::{Col, unzip, zip};

use crate::{
    E, SolverOptions,
    nlp::{NonlinearProgram, gd::Workspace},
    state::{SolverState, View},
};

/// Strategy for computing the step size at each gradient descent iteration.
pub trait StepSize {
    /// Creates a new step size strategy from the solver options.
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Computes the step size for the current iteration.
    fn compute(&mut self, state: &SolverState, work: &Workspace) -> E;
}

/// Constant step size: `α_k = learning_rate` for all `k`.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "learning_rate", type_ = E, description = "Constant learning rate for gradient descent.")]
pub struct ConstantStepSize {}

impl StepSize for ConstantStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn compute(&mut self, _state: &SolverState, _work: &Workspace) -> E {
        self.options.learning_rate
    }
}

/// Linear decay step size: `α_k = learning_rate / (1 + k)`.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "learning_rate", type_ = E, description = "Initial learning rate for linear decay step size.")]
pub struct LinearDecayStepSize {}

impl StepSize for LinearDecayStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn compute(&mut self, state: &SolverState, _work: &Workspace) -> E {
        self.options.learning_rate / (1. + state.nit() as E)
    }
}

/// Quadratic decay step size: `α_k = learning_rate / (1 + k²)`.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "learning_rate", type_ = E, description = "Initial learning rate for quadratic decay step size.")]
pub struct QuadraticDecayStepSize {}

impl StepSize for QuadraticDecayStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn compute(&mut self, state: &SolverState, _work: &Workspace) -> E {
        self.options.learning_rate / (1. + (state.nit() as E).powi(2))
    }
}

#[explicit_options(name = SolverOptions)]
pub struct BarzilaiBorweinStepSize {
    prev_x: Option<Col<E>>,
    prev_grad: Option<Col<E>>,
}

impl StepSize for BarzilaiBorweinStepSize {
    fn new(options: &SolverOptions) -> Self {
        Self {
            prev_x: None,
            prev_grad: None,

            options: options.into(),
        }
    }

    #[allow(non_snake_case)]
    fn compute(&mut self, state: &SolverState, work: &Workspace) -> E {
        let vars: &crate::state::Variables = state.variables();
        let step = if let (Some(prev_x), Some(prev_grad)) = (&self.prev_x, &self.prev_grad) {
            let dx: Col<E> = zip!(&vars.x(), prev_x).map(|unzip!(x_i, x_prev_i)| x_i - x_prev_i);
            let ddL: Col<E> =
                zip!(work.dL.as_ref(), prev_grad).map(|unzip!(g_i, g_prev_i)| g_i - g_prev_i);
            let numerator = zip!(&dx, &ddL)
                .map(|unzip!(dx_i, ddL_i)| dx_i * ddL_i)
                .sum();
            let denominator = zip!(&ddL, &ddL)
                .map(|unzip!(ddL_i, ddL_i2)| ddL_i * ddL_i2)
                .sum();
            if denominator.abs() > 1e-12 {
                numerator / denominator
            } else {
                1.
            }
        } else {
            1.
        };
        self.prev_x = Some(vars.x().to_owned());
        self.prev_grad = Some(work.dL.clone());

#[explicit_options(name = SolverOptions)]
#[use_option(name = "learning_rate", type_ = E, description = "Initial learning rate for Armijo line search step size.")]
#[use_option(name = "armijo_scale_factor", type_ = E, default = "0.5", description = "Backtracking line search parameter beta.")]
#[use_option(name = "armijo_parameter", type_ = E, default = "0.5", description = "Armijo condition parameter sigma.")]
pub struct ArmijoRule {
    alpha_prev: E,
}

impl ArmijoRule {
    fn compute_feasible(
        &self,
        nlp: &NonlinearProgram,
        state: &SolverState,
        work: &Workspace,
        m: E,
        t: E,
        alpha: E,
    ) -> E {
        let mut alpha = alpha;

        loop {
            let alpha_candidate = alpha / self.options.armijo_scale_factor;
            if alpha_candidate >= self.options.learning_rate
                || work.L - work.compute_L(nlp, state, alpha, 0.) <= alpha_candidate * t
            {
                break;
            }
            alpha = alpha_candidate;
        }

        alpha
    }

    fn compute_infeasible(
        &self,
        nlp: &NonlinearProgram,
        state: &SolverState,
        work: &Workspace,
        m: E,
        t: E,
        alpha: E,
    ) -> E {
        let mut alpha = alpha;

        loop {
            let alpha_candidate = alpha * self.options.armijo_scale_factor;
            if work.L - work.compute_L(nlp, state, alpha, 0.) >= alpha_candidate * t {
                break;
            }
            alpha = alpha_candidate;
        }

        alpha
    }
}

impl StepSize for ArmijoRule {
    fn new(options: &SolverOptions) -> Self {
        let options: ArmijoRuleInternalOptions = options.into();
        Self {
            alpha_prev: options.learning_rate,
            alpha_dual_prev: options.learning_rate,
            options,
        }
    }

    #[allow(non_snake_case)]
    fn compute<'a>(
        &mut self,
        nlp: &'a NonlinearProgram,
        state: &SolverState,
        work: &Workspace,
    ) -> (E, E) {
        let m = &work.dL.transpose() * &state.delta.dx();
        let t = -self.options.armijo_parameter * m;

        let alpha = self.alpha_prev;

        let alpha = {
            if work.L - work.compute_L(nlp, state, alpha, alpha) >= alpha * t {
                self.compute_feasible(nlp, state, work, m, t, alpha)
            } else {
                self.compute_infeasible(nlp, state, work, m, t, alpha)
            }
        };

        self.alpha_prev = alpha;

        (alpha, alpha)
    }
}
