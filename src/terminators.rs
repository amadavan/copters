//! Terminators for controlling and interrupting long-running processes.
//!
//! This module provides several implementations of the [`Terminator`] trait, including:
//! - [`InterruptTerminator`]: Responds to Ctrl-C (SIGINT) or programmatic interrupts.
//! - [`TimeOutTerminator`]: Terminates after a specified time limit.
//! - [`MultiTerminator`]: Combines multiple terminators.
//!
//! # Note
//! [`InterruptTerminator`] installs a global signal handler and **can only be constructed once** per process. Attempting to create multiple instances will result in a panic.

use std::sync::{Arc, atomic::AtomicBool};

use dyn_clone::DynClone;
use enum_dispatch::enum_dispatch;
use macros::{explicit_options, use_option};

use crate::{E, SolverOptions, SolverState, Status};

/// Criterion for deciding when the solver should stop.
///
/// Checked once per iteration. Returns `Some(Status)` to stop, or `None` to continue.
#[enum_dispatch]
pub trait Terminator: DynClone {
    /// Called once before the first iteration to reset any internal state (e.g. timers).
    fn init(&mut self, options: &SolverOptions);

    /// Returns `Some(status)` if the solver should stop, `None` otherwise.
    fn terminate(&mut self, state: &SolverState) -> Option<Status>;
}

/// A terminator that never triggers. The solver runs until the iteration limit.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NullTerminator {}

impl NullTerminator {
    pub fn new(_options: &SolverOptions) -> Self {
        Self {}
    }
}

impl Terminator for NullTerminator {
    fn init(&mut self, _options: &SolverOptions) {}

    fn terminate(&mut self, _state: &SolverState) -> Option<Status> {
        None
    }
}

/// Terminator that responds to Ctrl-C (SIGINT) or programmatic interrupts.
///
/// # Note
/// Only one instance of `InterruptTerminator` can be constructed per process, as it installs a global signal handler.
/// Creating more than one will panic.
#[derive(Clone)]
pub struct InterruptTerminator {
    interrupted: Arc<AtomicBool>,
}

impl InterruptTerminator {
    pub fn new(options: &SolverOptions) -> Self {
        Self {
            interrupted: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn interrupt(&mut self) {
        self.interrupted
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Terminator for InterruptTerminator {
    fn init(&mut self, _options: &SolverOptions) {
        let interrupted = Arc::new(AtomicBool::new(false));
        ctrlc::set_handler({
            let interrupted_clone = interrupted.clone();
            move || {
                interrupted_clone.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        })
        .expect("Error setting Ctrl-C handler");
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
#[derive(Clone)]
pub struct TimeOutTerminator {
    start_time: std::time::Instant,
}

impl TimeOutTerminator {
    pub fn new(options: &SolverOptions) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            options: options.into(),
        }
    }
}

impl Terminator for TimeOutTerminator {
    fn init(&mut self, options: &SolverOptions) {
        self.start_time = std::time::Instant::now();
        self.options = options.into();
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
#[derive(Clone)]
pub struct ConvergenceTerminator {}

impl ConvergenceTerminator {
    pub fn new(options: &SolverOptions) -> Self {
        Self {
            options: options.into(),
        }
    }
}

impl Terminator for ConvergenceTerminator {
    fn init(&mut self, options: &SolverOptions) {
        self.options = options.into();
    }

    fn terminate(&mut self, state: &SolverState) -> Option<Status> {
        if state.get_primal_infeasibility().norm_l2()
            <= self.options.tolerance * state.x.nrows() as E
            && state.get_dual_infeasibility().norm_l2()
                <= self.options.tolerance * state.y.nrows() as E
        {
            Some(Status::Optimal)
        } else {
            None
        }
    }
}

#[explicit_options(name = SolverOptions)]
#[use_option(name = "slow_progress_tolerance", type_ = E, default = "1e-8", description = "Tolerance for detecting slow progress in primal and dual infeasibility.")]
#[derive(Clone)]
pub struct SlowProgressTerminator {
    prev_state: Option<SolverState>,
}

impl SlowProgressTerminator {
    pub fn new(options: &SolverOptions) -> Self {
        Self {
            prev_state: None,
            options: options.into(),
        }
    }
}

impl Terminator for SlowProgressTerminator {
    fn init(&mut self, options: &SolverOptions) {
        self.options = options.into();
    }

    fn terminate(&mut self, state: &SolverState) -> Option<Status> {
        if let Some(prev) = &self.prev_state {
            let primal_diff =
                (state.get_primal_infeasibility() - prev.get_primal_infeasibility()).norm_l2();
            let dual_diff =
                (state.get_dual_infeasibility() - prev.get_dual_infeasibility()).norm_l2();
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

#[enum_dispatch(Terminator)]
#[derive(Clone)]
pub enum Terminators {
    NullTerminator(NullTerminator),
    InterruptTerminator(InterruptTerminator),
    TimeOutTerminator(TimeOutTerminator),
    ConvergenceTerminator(ConvergenceTerminator),
    SlowProgressTerminator(SlowProgressTerminator),
}

/// Combines multiple terminators; stops on the first one that fires.
#[derive(Clone)]
pub struct MultiTerminator {
    terminators: Vec<Terminators>,
}

impl MultiTerminator {
    pub fn new(terminators: Vec<Terminators>) -> Self {
        Self { terminators }
    }

    pub fn new_empty() -> Self {
        Self {
            terminators: Vec::new(),
        }
    }

    pub fn add_terminator(&mut self, terminator: Terminators) {
        self.terminators.push(terminator);
    }
}

impl Terminator for MultiTerminator {
    fn init(&mut self, options: &SolverOptions) {
        for terminator in &mut self.terminators {
            terminator.init(options);
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

#[allow(unused)]
struct Builder {
    terminators: Vec<Terminators>,
    options: SolverOptions,
}

#[allow(unused)]
impl Builder {
    pub fn new() -> Self {
        Self {
            terminators: Vec::new(),
            options: SolverOptions::new(),
        }
    }

    pub fn with_options(mut self, options: SolverOptions) -> Self {
        self.options = options;
        self
    }

    pub fn add_terminator(mut self, terminator: Terminators) -> Self {
        self.terminators.push(terminator);
        self
    }

    pub fn build(self) -> Box<dyn Terminator> {
        Box::new(MultiTerminator::new(self.terminators.into_iter().collect()))
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
    #[ignore]
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
