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

use macros::{build_option_enum, explicit_options, use_option};

use crate::{E, SolverOptions, SolverState, Status};

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
        if state.get_primal_infeasibility() <= self.options.tolerance * state.x.nrows() as E
            && state.get_dual_infeasibility() <= self.options.tolerance * state.y.nrows() as E
        {
            Some(Status::Optimal)
        } else {
            None
        }
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

build_option_enum!(
    trait_ = Terminator,
    name = "Terminators",
    variants = (
        NullTerminator,
        InterruptTerminator,
        TimeOutTerminator,
        ConvergenceTerminator,
        MultiTerminator
    ),
    new_arguments = (&SolverOptions,),
    doc_header = "Termination criteria for the solver."
);

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
