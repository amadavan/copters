//! Terminators for controlling and interrupting long-running processes.
//!
//! This module provides several implementations of the [`Terminator`] trait, including:
//! - [`InterruptTerminator`]: Responds to Ctrl-C (SIGINT) or programmatic interrupts.
//! - [`TimeOutTerminator`]: Terminates after a specified time limit.
//! - [`MultiTerminator`]: Combines multiple terminators.
//!
//! # Note
//! [`InterruptTerminator`] installs a global signal handler and **can only be constructed once** per process. Attempting to create multiple instances will result in a panic.

use std::{
    collections::HashSet,
    sync::{Arc, atomic::AtomicBool},
};

use macros::{build_option_enum, explicit_options, use_option};

use crate::{E, Solver, SolverOptions, SolverState, Status};

/// Criterion for deciding when the solver should stop.
///
/// Checked once per iteration. Returns `Some(Status)` to stop, or `None` to continue.
pub trait Terminator {
    /// Creates a new terminator from solver options.
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    /// Called once before the first iteration to reset any internal state (e.g. timers).
    fn initialize(&mut self) {}

    /// Returns `Some(status)` if the solver should stop, `None` otherwise.
    fn terminate(&mut self, state: &SolverState) -> Option<Status>;
}

/// A terminator that never triggers. The solver runs until the iteration limit.
pub struct NullTerminator {}

impl Terminator for NullTerminator {
    fn new(_options: &SolverOptions) -> Self {
        Self {}
    }

    fn terminate(&mut self, _state: &SolverState) -> Option<Status> {
        None
    }
}

/// Terminator that responds to Ctrl-C (SIGINT) or programmatic interrupts.
///
/// # Note
/// Only one instance of `InterruptTerminator` can be constructed per process, as it installs a global signal handler.
/// Creating more than one will panic.
pub struct InterruptTerminator {
    interrupted: Arc<AtomicBool>,
}

impl InterruptTerminator {
    pub fn interrupt(&mut self) {
        self.interrupted
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Terminator for InterruptTerminator {
    fn new(_options: &SolverOptions) -> Self {
        let interrupted = Arc::new(AtomicBool::new(false));
        ctrlc::set_handler({
            let interrupted_clone = interrupted.clone();
            move || {
                interrupted_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        })
        .expect("Error setting Ctrl-C handler");
        Self { interrupted }
    }

    fn terminate(&mut self, _state: &SolverState) -> Option<Status> {
        if self.interrupted.load(std::sync::atomic::Ordering::SeqCst) {
            Some(Status::Interrupted)
        } else {
            None
        }
    }
}

/// Terminator that triggers after a specified number of seconds.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "max_time", type_ = u64, default = "3600", description = "Maximum time in seconds before termination")]
pub struct TimeOutTerminator {
    start_time: std::time::Instant,
}

impl TimeOutTerminator {}

impl Terminator for TimeOutTerminator {
    fn new(options: &SolverOptions) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            options: options.into(),
        }
    }

    fn initialize(&mut self) {
        self.start_time = std::time::Instant::now();
    }

    fn terminate(&mut self, _state: &SolverState) -> Option<Status> {
        if self.start_time.elapsed().as_secs() >= self.options.max_time {
            Some(Status::TimeLimit)
        } else {
            None
        }
    }
}

/// Terminates when both primal and dual infeasibility fall below `tolerance`.
#[explicit_options(name = SolverOptions)]
#[use_option(name = "tolerance", type_ = E, default = "1e-7", description = "Tolerance for convergence-based termination")]
pub struct ConvergenceTerminator {}

impl Terminator for ConvergenceTerminator {
    fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }

    fn terminate(&mut self, state: &SolverState) -> Option<Status> {
        if state.get_primal_infeasibility() <= self.options.tolerance
            && state.get_dual_infeasibility() <= self.options.tolerance
        {
            Some(Status::Optimal)
        } else {
            None
        }
    }
}

#[explicit_options(name = SolverOptions)]
#[use_option(name = "slow_progress_tolerance", type_ = E, default = "1e-8", description = "Tolerance for detecting slow progress in primal and dual infeasibility.")]
pub struct SlowProgressTerminator {
    prev_state: Option<SolverState>,
}

impl Terminator for SlowProgressTerminator {
    fn new(options: &SolverOptions) -> Self {
        Self {
            prev_state: None,
            options: options.into(),
        }
    }

    fn terminate(&mut self, state: &SolverState) -> Option<Status> {
        if let Some(prev) = &self.prev_state {
            let primal_diff =
                (state.get_primal_infeasibility() - prev.get_primal_infeasibility()).abs();
            let dual_diff = (state.get_dual_infeasibility() - prev.get_dual_infeasibility()).abs();
            if primal_diff <= self.options.slow_progress_tolerance
                && dual_diff <= self.options.slow_progress_tolerance
            {
                return Some(Status::Optimal);
            }
        }
        self.prev_state = Some(state.clone());
        None
    }
}

/// Combines multiple terminators; stops on the first one that fires.
pub struct MultiTerminator {
    terminators: Vec<Box<dyn Terminator>>,
}

impl MultiTerminator {
    pub fn new_default(options: &SolverOptions) -> Self {
        Self {
            terminators: vec![
                Box::new(InterruptTerminator::new(&options)),
                Box::new(TimeOutTerminator::new(&options)),
                Box::new(ConvergenceTerminator::new(&options)),
            ],
        }
    }

    pub fn new_with_terminators(terminators: Vec<Box<dyn Terminator>>) -> Self {
        Self { terminators }
    }

    pub fn add_terminator(&mut self, terminator: Box<dyn Terminator>) {
        self.terminators.push(terminator);
    }
}

impl Terminator for MultiTerminator {
    fn new(options: &SolverOptions) -> Self {
        Self::new_default(options)
    }

    fn initialize(&mut self) {
        for terminator in &mut self.terminators {
            terminator.initialize();
        }
    }

    fn terminate(&mut self, state: &SolverState) -> Option<Status> {
        for terminator in &mut self.terminators {
            if let Some(status) = terminator.terminate(state) {
                return Some(status);
            }
        }
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerminatorType {
    NullTerminator,
    InterruptTerminator,
    TimeOutTerminator,
    ConvergenceTerminator,
    SlowProgressTerminator,
    MultiTerminator,
}

#[allow(unused)]
struct Builder {
    terminators: HashSet<TerminatorType>,
    options: SolverOptions,
}

impl Builder {
    pub fn new() -> Self {
        Self {
            terminators: HashSet::new(),
            options: SolverOptions::new(),
        }
    }

    pub fn with_options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn add_terminator(mut self, terminator: TerminatorType) -> Self {
        self.terminators.insert(terminator);
        self
    }

    pub fn build(self) -> MultiTerminator {
        let terminators = self
            .terminators
            .into_iter()
            .map(|t| match t {
                TerminatorType::NullTerminator => {
                    Box::new(NullTerminator::new(&self.options)) as Box<dyn Terminator>
                }
                TerminatorType::InterruptTerminator => {
                    Box::new(InterruptTerminator::new(&self.options))
                }
                TerminatorType::TimeOutTerminator => {
                    Box::new(TimeOutTerminator::new(&self.options))
                }
                TerminatorType::ConvergenceTerminator => {
                    Box::new(ConvergenceTerminator::new(&self.options))
                }
                TerminatorType::SlowProgressTerminator => {
                    Box::new(SlowProgressTerminator::new(&self.options))
                }
                TerminatorType::MultiTerminator => {
                    Box::new(MultiTerminator::new_default(&self.options))
                }
            })
            .collect();
        MultiTerminator::new_with_terminators(terminators)
    }
}

#[cfg(test)]
mod tests {
    use faer::col::generic::Col;

    use super::*;

    #[cfg(unix)]
    mod platform {
        pub fn send_sigint() {
            use nix::sys::signal::{self, Signal};
            use nix::unistd::Pid;

            let pid = std::process::id() as i32;
            signal::kill(Pid::from_raw(pid), Signal::SIGINT).expect("Failed to send SIGINT");
        }
    }

    #[cfg(windows)]
    mod platform {
        pub fn send_sigint() {
            use winapi::um::wincon::CTRL_C_EVENT;
            use winapi::um::wincon::GenerateConsoleCtrlEvent;

            unsafe {
                GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0);
            }
        }
    }

    #[test]
    fn test_interruption_terminator_ctrlc() {
        let options = SolverOptions::new();
        let mut terminator = InterruptTerminator::new(&options);
        let state = SolverState::new(Col::zeros(0), Col::zeros(0), Col::zeros(0), Col::zeros(0)); // Dummy state

        std::thread::spawn(|| {
            std::thread::sleep(std::time::Duration::from_secs(2));
            platform::send_sigint();
        });

        loop {
            if let Some(status) = terminator.terminate(&state) {
                assert_eq!(status, Status::Interrupted);
                break;
            }
        }
    }
}
