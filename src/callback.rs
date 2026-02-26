use std::{collections::HashSet, fmt::Debug};

use dyn_clone::DynClone;
use enum_dispatch::enum_dispatch;

use crate::{E, SolverOptions, SolverState};

/// Hook invoked once per solver iteration for logging, monitoring, or early stopping.
#[enum_dispatch]
pub trait Callback: Debug + DynClone {
    fn init(&mut self, _state: &SolverState) {}

    /// Called at the end of each iteration with the current solver state.
    fn call(&mut self, _state: &SolverState) {}

    fn finish(&mut self) {}
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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
            state.get_primal_infeasibility().norm_l2() / state.x.nrows() as E,
            state.get_dual_infeasibility().norm_l2() / state.x.nrows() as E,
            state.get_complimentary_slack_lower().norm_l2() / state.x.nrows() as E,
            state.get_complimentary_slack_upper().norm_l2() / state.x.nrows() as E,
        );
        println!("{txt}");
    }

    fn finish(&mut self) {
        println!("");
    }
}

#[enum_dispatch(Callback)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Callbacks {
    NoOp(NoOpCallback),
    ConvergenceOutput(ConvergenceOutput),
}

#[derive(Debug, Clone)]
struct MultiCallback {
    callbacks: Vec<Callbacks>,
}

impl MultiCallback {
    pub fn new(callbacks: Vec<Callbacks>) -> Self {
        Self { callbacks }
    }

    pub fn new_empty() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

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

    fn finish(&mut self) {
        for cb in &mut self.callbacks {
            <Callbacks as Callback>::finish(cb);
        }
    }
}

pub struct Builder {
    callback: HashSet<Callbacks>,
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

    pub fn add_callback(mut self, callback: Callbacks) -> Self {
        self.callback.insert(callback);
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
