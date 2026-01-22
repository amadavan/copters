
//! Terminators for controlling and interrupting long-running processes.
//!
//! This module provides several implementations of the [`Terminator`] trait, including:
//! - [`InterruptTerminator`]: Responds to Ctrl-C (SIGINT) or programmatic interrupts.
//! - [`TimeOutTerminator`]: Terminates after a specified time limit.
//! - [`MultipleTerminators`]: Combines multiple terminators.
//!
//! # Note
//! [`InterruptTerminator`] installs a global signal handler and **can only be constructed once** per process. Attempting to create multiple instances will result in a panic.

use std::sync::{Arc, atomic::AtomicBool};

use crate::Status;


pub trait Terminator {
  fn initialize(&mut self) {}

  fn terminate(&mut self) -> Option<Status>;
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
  pub fn new() -> Self {
    let interrupted = Arc::new(AtomicBool::new(false));
    ctrlc::set_handler({
      let interrupted_clone = interrupted.clone();
      move || {
        interrupted_clone.store(true, std::sync::atomic::Ordering::SeqCst);
      }
    }).expect("Error setting Ctrl-C handler");
    Self { interrupted }
  }

  pub fn interrupt(&mut self) {
    self.interrupted.store(true, std::sync::atomic::Ordering::SeqCst);
  }
}

impl Terminator for InterruptTerminator {
  fn terminate(&mut self) -> Option<Status> {
    if self.interrupted.load(std::sync::atomic::Ordering::SeqCst) {
      Some(Status::Interrupted)
    } else {
      None
    }
  }
}


/// Terminator that triggers after a specified number of seconds.
pub struct TimeOutTerminator {
  max_time_secs: u64,
  start_time: std::time::Instant,
}

impl TimeOutTerminator {
  pub fn new(max_time_secs: u64) -> Self {
    Self {
      max_time_secs,
      start_time: std::time::Instant::now(),
    }
  }
}

impl Terminator for TimeOutTerminator {
  fn initialize(&mut self) {
    self.start_time = std::time::Instant::now();
  }

  fn terminate(&mut self) -> Option<Status> {
    if self.start_time.elapsed().as_secs() >= self.max_time_secs {
      Some(Status::TimeLimit)
      } else {
        None
      }
  }
}



/// Terminator that combines multiple terminators and triggers if any of them do.
pub struct MultipleTerminators {
  terminators: Vec<Box<dyn Terminator>>,
}

impl MultipleTerminators {
  pub fn new(terminators: Vec<Box<dyn Terminator>>) -> Self {
    Self { terminators }
  }
}

impl Terminator for MultipleTerminators {
  fn initialize(&mut self) {
    for terminator in &mut self.terminators {
      terminator.initialize();
    }
  }

  fn terminate(&mut self) -> Option<Status> {
    for terminator in &mut self.terminators {
      if let Some(status) = terminator.as_mut().terminate() {
        return Some(status);
      }
    }
    None
  }
}

#[cfg(test)]
mod tests {
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
      use winapi::um::wincon::GenerateConsoleCtrlEvent;
      use winapi::um::wincon::CTRL_C_EVENT;

      unsafe {
        GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0);
      }
    }
  }

  #[test]
  fn test_interruption_terminator_ctrlc() {
    let mut terminator = InterruptTerminator::new();

  std::thread::spawn(|| {
    std::thread::sleep(std::time::Duration::from_secs(2));
    platform::send_sigint();
  });

    println!("Press Ctrl-C to test interruption terminator...");
    loop {
      if let Some(status) = terminator.terminate() {
        assert_eq!(status, Status::Interrupted);
        break;
      }
    }
  }
}