use std::backtrace::Backtrace;

use faer::dyn_stack::mem::AllocError;
use faer::linalg::solvers::LdltError;
use faer::sparse::linalg::LuError;
use faer::sparse::{FaerError, SparseColMatRef};
use faer::{Mat, MatMut, MatRef};
use snafu::Snafu;

use crate::{E, I};

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(super)))]
pub enum SolverError {
    #[snafu(display("Symbolic factorization error: {}", message))]
    SymbolicFactorization {
        message: String,
        source: FaerError,
        backtrace: Backtrace,
    },

    #[snafu(display("Cholesky factorization error: {}", message))]
    CholeskyFactorization {
        message: String,
        source: LdltError,
        backtrace: Backtrace,
    },

    #[snafu(display("LU factorization error: {}", message))]
    LuFactorization {
        message: String,
        source: LuError,
        backtrace: Backtrace,
    },

    #[snafu(display("Numeric factorization error"))]
    NumericFactorization { backtrace: Backtrace },

    #[snafu(display("{message} has not be initialized."))]
    Uninitialized {
        message: String,
        backtrace: Backtrace,
    },

    #[snafu(display("Memory reservation failed"))]
    MemoryReservation {
        source: std::collections::TryReserveError,
        backtrace: Backtrace,
    },

    #[snafu(display("Memory allocation failed"))]
    MemoryAllocation {
        source: AllocError,
        backtrace: Backtrace,
    },

    #[snafu(display("Unable to solve linear system"))]
    SolveFailed { backtrace: Backtrace },
}

/// Trait for symmetric linear solvers supporting matrix analysis, factorization, and solving linear
/// systems.
///
/// This trait provides a standard interface for working with sparse matrices and right-hand side
/// vectors. Implementors must call `analyze` and `factorize` before solving systems.
pub trait Solver {
    fn new() -> Self
    where
        Self: Sized;

    /// Performs symbolic analysis of the given sparse matrix and prepares for factorization.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn analyze(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), SolverError>;

    /// Performs numeric factorization of the matrix after symbolic analysis.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn factorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), SolverError>;

    /// Refactorizes the matrix, typically used when the matrix structure remains but values change.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn refactorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), SolverError>;

    /// Solves the linear system in place for the given right-hand side vector `b`.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn solve_in_place(&self, b: &mut MatMut<E>) -> Result<(), SolverError>;

    /// Solves the linear system for the given right-hand side vector `b` and returns the solution
    /// matrix. Returns the solution matrix on success, or an error message on failure.
    fn solve(&self, b: MatRef<E>) -> Result<Mat<E>, SolverError>;
}
