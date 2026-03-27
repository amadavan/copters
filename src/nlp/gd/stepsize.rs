use std::str::FromStr;

use macros::{explicit_options, use_option};

use faer::{Col, ColRef};

use crate::{
    E, OptionTrait, SolverOptions,
    nlp::{NonlinearProgram, gd::Workspace},
    state::SolverState,
};

/// Strategy for computing the step size at each gradient descent iteration.
pub trait StepSize {
    /// Creates a new step size strategy from the solver options.
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Computes the step size for the current iteration.
    fn compute<'a>(
        &mut self,
        nlp: &'a NonlinearProgram,
        state: &SolverState,
        work: &Workspace,
    ) -> (E, E);
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

    fn compute<'a>(
        &mut self,
        _nlp: &'a NonlinearProgram,
        _state: &SolverState,
        _work: &Workspace,
    ) -> (E, E) {
        (self.options.learning_rate, self.options.learning_rate)
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

    fn compute<'a>(
        &mut self,
        _nlp: &'a NonlinearProgram,
        state: &SolverState,
        _work: &Workspace,
    ) -> (E, E) {
        let step = self.options.learning_rate / (1. + state.nit() as E);
        (step, step)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BarzilaiBorweinVariant {
    ShortStep,
    LongStep,
    Hybrid,
}

impl OptionTrait for BarzilaiBorweinVariant {}

impl FromStr for BarzilaiBorweinVariant {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "short" => Ok(Self::ShortStep),
            "long" => Ok(Self::LongStep),
            "hybrid" => Ok(Self::Hybrid),
            _ => Err(format!("Invalid Barzilai-Borwein variant: {}", s)),
        }
    }
}

#[explicit_options(name = SolverOptions)]
#[use_option(name = "alpha_min", type_ = E, default = "1e-10", description = "Minimum step size for Barzilai-Borwein step size.")]
#[use_option(name = "alpha_max", type_ = E, default = "1e10", description = "Maximum step size for Barzilai-Borwein step size.")]
#[use_option(name = "learning_rate", type_ = E, description = "Initial learning rate for Barzilai-Borwein step size.")]
#[use_option(name = "barzilai_borwein_variant", type_ = crate::nlp::gd::stepsize::BarzilaiBorweinVariant, default = "short", description = "Variant of the Barzilai-Borwein step size to use (short, long, or hybrid).")]
pub struct BarzilaiBorwein {
    prev_x: Option<Col<E>>,
    prev_grad: Option<Col<E>>,
    prev_y: Option<Col<E>>,
    prev_g: Option<Col<E>>,

    fallback: ArmijoRule,
}

impl BarzilaiBorwein {
    fn compute_step(&self, dx: ColRef<E>, df: ColRef<E>) -> E {
        match self.options.barzilai_borwein_variant {
            BarzilaiBorweinVariant::ShortStep => {
                let numerator = &dx.transpose() * &df;
                let denominator = &df.transpose() * &df;
                if denominator.abs() > 1e-12 {
                    numerator / denominator
                } else {
                    0.
                }
            }
            BarzilaiBorweinVariant::LongStep => {
                let numerator = &dx.transpose() * &dx;
                let denominator = &dx.transpose() * &df;
                if denominator.abs() > 1e-12 {
                    numerator / denominator
                } else {
                    0.
                }
            }
            BarzilaiBorweinVariant::Hybrid => {
                // Hybrid step size: choose the short or long step based on the curvature condition.
                let curvature_condition =
                    (&dx.transpose() * &df).abs() / (&dx.transpose() * &dx).abs();
                if curvature_condition < 0.5 {
                    // Short step
                    let numerator = &dx.transpose() * &df;
                    let denominator = &df.transpose() * &df;
                    if denominator.abs() > 1e-12 {
                        numerator / denominator
                    } else {
                        0.
                    }
                } else {
                    // Long step
                    let numerator = &dx.transpose() * &dx;
                    let denominator = &dx.transpose() * &df;
                    if denominator.abs() > 1e-12 {
                        numerator / denominator
                    } else {
                        0.
                    }
                }
            }
        }
    }
}

impl StepSize for BarzilaiBorwein {
    fn new(options: &SolverOptions) -> Self {
        Self {
            prev_x: None,
            prev_grad: None,
            prev_y: None,
            prev_g: None,
            fallback: ArmijoRule::new(options),

            options: options.into(),
        }
    }

    #[allow(non_snake_case)]
    fn compute<'a>(
        &mut self,
        _nlp: &'a NonlinearProgram,
        state: &SolverState,
        work: &Workspace,
    ) -> (E, E) {
        let vars: &crate::state::Variables = state.variables();
        let alpha_primal = if let (Some(prev_x), Some(prev_grad)) = (&self.prev_x, &self.prev_grad)
        {
            let dx = vars.x() - prev_x;
            let ddL = work.dL.as_ref() - prev_grad;

            self.compute_step(dx.as_ref(), ddL.as_ref())
        } else {
            0.
        };

        let alpha_dual = if let (Some(prev_y), Some(prev_g)) = (&self.prev_y, &self.prev_g) {
            let dy = vars.y() - prev_y;
            let ddg = work.g.as_ref() - prev_g;

            self.compute_step(dy.as_ref(), ddg.as_ref())
        } else {
            0.
        };

        self.prev_x = Some(vars.x().to_owned());
        self.prev_grad = Some(work.dL.clone());
        self.prev_y = Some(vars.y().to_owned());
        self.prev_g = Some(work.g.clone());

        if alpha_primal <= 0. || alpha_primal.is_nan() {
            return self.fallback.compute(_nlp, state, work);
        }
        if alpha_dual <= 0. || alpha_dual.is_nan() {
            return self.fallback.compute(_nlp, state, work);
        }

        (alpha_primal, alpha_primal)
    }
}

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
                self.compute_feasible(nlp, state, work, t, alpha)
            } else {
                self.compute_infeasible(nlp, state, work, t, alpha)
            }
        };

        self.alpha_prev = alpha;

        (alpha, alpha)
    }
}
