use derive_more::{Display, Error};
use faer::sparse::SparseColMatRef;
use faer::{Mat, MatMut, MatRef};
use problemo::Problem;

use crate::{E, I};

#[derive(Debug, Display, Error, PartialEq)]
pub enum LinearSolverError {
    #[display("Symbolic factorization error")]
    SymbolicFactorization,

    #[display("Cholesky factorization error")]
    CholeskyFactorization,

    #[display("LU factorization error")]
    LuFactorization,

    #[display("Numeric factorization error")]
    NumericFactorization,

    #[display("Uninitialized error")]
    Uninitialized,

    #[display("Memory reservation failed")]
    MemoryReservation,

    #[display("Memory allocation failed")]
    MemoryAllocation,

    #[display("Unable to solve linear system")]
    SolveFailed,
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
    fn analyze(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem>;

    /// Performs numeric factorization of the matrix after symbolic analysis.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn factorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem>;

    /// Refactorizes the matrix, typically used when the matrix structure remains but values change.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn refactorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem>;

    /// Solves the linear system in place for the given right-hand side vector `b`.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn solve_in_place(&self, b: &mut MatMut<E>) -> Result<(), Problem>;

    /// Solves the linear system for the given right-hand side vector `b` and returns the solution
    /// matrix. Returns the solution matrix on success, or an error message on failure.
    fn solve(&self, b: MatRef<E>) -> Result<Mat<E>, Problem>;
}

pub trait LinearSolver: Solver {}

/// Trait for symmetric linear solvers supporting matrix analysis, factorization, and solving linear
/// systems.
///
/// This trait provides a standard interface for working with sparse matrices and right-hand side
/// vectors. Implementors must call `analyze` and `factorize` before solving systems.
pub trait SymmetricLinearSolver: LinearSolver {}

#[cfg(test)]
mod tests {
    use crate::linalg::{
        cholesky::{SimplicialSparseCholesky, SupernodalSparseCholesky},
        lu::SimplicialSparseLu,
        solver::{LinearSolver, SymmetricLinearSolver},
    };
    use faer::rand::SeedableRng;
    use faer::rand::rngs::StdRng;
    use faer::stats::DistributionExt;
    use faer::stats::prelude::{CwiseMatDistribution, StandardNormal};
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn test_mtx_symsolver(
        #[values("Trefethen 20b")] mat_name: &str,
        #[values(
            SimplicialSparseCholesky::new(),
            SupernodalSparseCholesky::new(),
            SimplicialSparseLu::new()
        )]
        mut solver: impl LinearSolver,
    ) {
        const N_COUNT: usize = 10;
        let mat = loaders::mtx::get_matrix_by_name::<I, E>(mat_name, true);

        solver.analyze(mat.as_ref()).unwrap();
        solver.factorize(mat.as_ref()).unwrap();

        let rng = &mut StdRng::seed_from_u64(0);

        let n = mat.ncols();

        // Generate several random column vectors and verify whether the results hold
        for _ in 0..N_COUNT {
            // let mut col: Mat<E> = Mat::zeros(n, 1);
            let col = CwiseMatDistribution {
                nrows: n,
                ncols: 1,
                dist: StandardNormal,
            }
            .rand(rng);

            let mut result = col.clone();
            result = solver.solve(result.as_ref()).expect("Unable to solve");

            // println!("SolverError: {:e}", (&col - &mat * &result).norm_l2());
            assert!((&col - &mat * &result).norm_l2() < 1e-10); // Check if Ax ≈ b
        }
    }
}
