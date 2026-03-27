use std::{fmt::Debug, time::Instant};

use dyn_clone::DynClone;
use enum_dispatch::enum_dispatch;
use macros::{explicit_options, use_option};

use crate::{E, SolverOptions, state::SolverState};

/// Hook invoked once per solver iteration for logging, monitoring, or early stopping.
#[enum_dispatch]
pub trait Callback: DynClone {
    fn init(&mut self, _state: &SolverState) {}

    /// Called at the end of each iteration with the current solver state.
    fn call(&mut self, _state: &SolverState) {}

    fn finish(&mut self, _state: &SolverState) {}
}

/// A callback that does nothing. Use when no per-iteration output is needed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NoOpCallback {}

impl NoOpCallback {
    pub fn new() -> Self {
        Self {}
    }
}

impl Callback for NoOpCallback {}

/// Prints primal and dual infeasibility to stdout each iteration.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "output_timeout", type_ = f64, default = "0.0", description = "Minimum time (in seconds) between output lines. Set to 0 for no throttling.")]
#[use_option(name = "frequency", type_ = usize, default = "1", description = "Print output every N iterations.")]
#[derive(Clone)]
pub struct ConvergenceOutput {
    otime: Option<Instant>,
}

impl ConvergenceOutput {
    pub fn new(options: &SolverOptions) -> Self {
        Self {
            otime: None,
            options: options.into(),
        }
    }
}

impl Callback for ConvergenceOutput {
    fn init(&mut self, _state: &SolverState) {
        let header = format!(
            "| {:5} | {:8} | {:8} | {:8} | {:8} | {:8} | {:8} |",
            "NIT", "D_PRIMAL", "D_DUAL", "PRI_INF", "DUAL_INF", "CS_L", "CS_U"
        );

        let separator = "-".repeat(header.len());
        println!("");
        println!("{header}");
        println!("{separator}");

        self.otime = None;
    }

    fn call(&mut self, state: &SolverState) {
        // Check for timeout
        if let Some(otime) = self.otime {
            let elapsed = otime.elapsed();
            if elapsed.as_secs_f64() < self.options.output_timeout {
                return;
            }
        }

        // Check for frequency
        if state.nit() % self.options.frequency != 0 {
            return;
        }

        let residual = state.residuals();
        let (n, m) = (residual.dual().nrows(), residual.primal().nrows());
        let txt = format!(
            "| {:5} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} |",
            state.nit(),
            state.alpha_primal(),
            state.alpha_dual(),
            residual.primal().norm_l2() / m as E,
            residual.dual().norm_l2() / n as E,
            residual.slack_l().norm_l2() / n as E,
            residual.slack_u().norm_l2() / n as E,
        );
        println!("{txt}");
        self.otime = Some(Instant::now());
    }

    fn finish(&mut self, state: &SolverState) {
        let residual = state.residuals();
        let (n, m) = (residual.dual().nrows(), residual.primal().nrows());
        let txt = format!(
            "| {:5} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} |",
            state.nit(),
            state.alpha_primal(),
            state.alpha_dual(),
            residual.primal().norm_l2() / m as E,
            residual.dual().norm_l2() / n as E,
            residual.slack_l().norm_l2() / n as E,
            residual.slack_u().norm_l2() / n as E,
        );
        println!("{txt}");
        println!("");
    }
}

#[enum_dispatch(Callback)]
#[derive(Clone)]
pub enum Callbacks {
    NoOp(NoOpCallback),
    ConvergenceOutput(ConvergenceOutput),
}

#[derive(Clone)]
struct MultiCallback {
    callbacks: Vec<Callbacks>,
}

impl MultiCallback {
    pub fn new(callbacks: Vec<Callbacks>) -> Self {
        Self { callbacks }
    }

    #[allow(unused)]
    pub fn new_empty() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    #[allow(unused)]
    pub fn add_callback(&mut self, callback: Callbacks) {
        self.callbacks.push(callback);
    }
}

impl Callback for MultiCallback {
    fn init(&mut self, state: &SolverState) {
        for cb in &mut self.callbacks {
            <Callbacks as Callback>::init(cb, state);
        }
    }

    fn call(&mut self, state: &SolverState) {
        for cb in &mut self.callbacks {
            <Callbacks as Callback>::call(cb, state);
        }
    }

    fn finish(&mut self, state: &SolverState) {
        for cb in &mut self.callbacks {
            <Callbacks as Callback>::finish(cb, state);
        }
    }
}

pub struct Builder {
    callback: Vec<Callbacks>,
    options: SolverOptions,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            callback: Vec::new(),
            options: SolverOptions::new(),
        }
    }

    pub fn with_options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn add_callback(mut self, callback: Callbacks) -> Self {
        self.callback.push(callback);
        self
    }

    pub fn build(&self) -> Box<dyn Callback> {
        if self.callback.len() == 0 {
            return Box::new(NoOpCallback::new());
        } else if self.callback.len() == 1 {
            return Box::new(self.callback.iter().next().unwrap().clone());
        }
        Box::new(MultiCallback::new(self.callback.iter().cloned().collect()))
    }
}
