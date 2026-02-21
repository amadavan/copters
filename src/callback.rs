use std::collections::HashSet;

use crate::{SolverOptions, SolverState};

/// Hook invoked once per solver iteration for logging, monitoring, or early stopping.
pub trait Callback {
    fn init(&mut self, _state: &SolverState) {}

    /// Called at the end of each iteration with the current solver state.
    fn call(&mut self, _state: &SolverState) {}

    fn finish(&mut self) {}
}

/// A callback that does nothing. Use when no per-iteration output is needed.
pub struct NoOpCallback {}

impl NoOpCallback {
    pub fn new() -> Self {
        Self {}
    }
}

impl Callback for NoOpCallback {}
/// Prints primal and dual infeasibility to stdout each iteration.
pub struct ConvergenceOutput {}

impl ConvergenceOutput {
    pub fn new() -> Self {
        Self {}
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
    }

    fn call(&mut self, state: &SolverState) {
        let txt = format!(
            "| {:5} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} | {:<8.2e} |",
            state.nit,
            state.alpha_primal,
            state.alpha_dual,
            state.get_primal_infeasibility(),
            state.get_dual_infeasibility(),
            state.get_complimentary_slack_lower(),
            state.get_complimentary_slack_upper(),
        );
        println!("{txt}");
    }

    fn finish(&mut self) {
        println!("");
    }
}

pub struct MultiCallback {
    callbacks: Vec<Box<dyn Callback>>,
}

impl MultiCallback {
    pub fn new(callbacks: Vec<Box<dyn Callback>>) -> Self {
        Self { callbacks }
    }
}

impl Callback for MultiCallback {
    fn init(&mut self, state: &SolverState) {
        for cb in &mut self.callbacks {
            cb.init(state);
        }
    }

    fn call(&mut self, state: &SolverState) {
        for cb in &mut self.callbacks {
            cb.call(state);
        }
    }

    fn finish(&mut self) {
        for cb in &mut self.callbacks {
            cb.finish();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallbackType {
    ConvergenceOutput,
}

pub struct Builder {
    callback: HashSet<CallbackType>,
    options: SolverOptions,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            callback: HashSet::new(),
            options: SolverOptions::new(),
        }
    }

    pub fn with_options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn add_callback(mut self, callback: CallbackType) -> Self {
        self.callback.insert(callback);
        self
    }

    pub fn build(&self) -> Box<dyn Callback> {
        let callbacks: Vec<Box<dyn Callback>> = self
            .callback
            .iter()
            .map(|cb_type| match cb_type {
                CallbackType::ConvergenceOutput => {
                    Box::new(ConvergenceOutput::new()) as Box<dyn Callback>
                }
            })
            .collect();

        Box::new(MultiCallback::new(callbacks))
    }
}
