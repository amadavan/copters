//! # Linear Solvers for General Sparse Matrices using LU Factorization
//!
//! This module provides traits and implementations for solving general sparse linear systems
//! using LU factorization with the Faer library. It includes:
//! - The [`SimplicialSparseLu`] solver for simplicial LU factorization.
//! - Support for both symmetric and non-symmetric matrices.
//! - Unit tests for correctness and accuracy.
//!
//! ## Example Usage
//! ```
//! use faer::sparse::{SparseColMat, Triplet};
//! use copters::linalg::lu::SimplicialSparseLu;
//! use copters::linalg::solver::Solver;
//!
//! // Construct a general sparse matrix
//! let n = 3;
//! let triplets = vec![
//!     Triplet::new(0, 0, 4.0),
//!     Triplet::new(0, 1, -1.0),
//!     Triplet::new(1, 0, -1.0),
//!     Triplet::new(1, 1, 4.0),
//!     Triplet::new(1, 2, -1.0),
//!     Triplet::new(2, 1, -1.0),
//!     Triplet::new(2, 2, 4.0),
//! ];
//! let mat = SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();
//!
//! // Create and use the solver
//! let mut solver = SimplicialSparseLu::new();
//! solver.analyze(mat.as_ref()).unwrap();
//! solver.factorize(mat.as_ref()).unwrap();
//! let b = faer::Mat::from_fn(n, 1, |i, _| (i + 1) as f64);
//! let x = solver.solve(b.as_ref()).unwrap();
//! ```

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::perm::Perm;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::sparse::linalg::colamd;
use faer::sparse::linalg::lu::simplicial::{self, SimplicialLu};
use faer::sparse::SparseColMatRef;
use faer::{Mat, MatMut, MatRef};
use snafu::prelude::*;

use crate::linalg::solver::{
    LuFactorizationSnafu, MemoryAllocationSnafu, MemoryReservationSnafu, Solver, SolverError,
    SymbolicFactorizationSnafu, UninitializedSnafu,
};
use crate::{E, I};

/// Sparse LU solver using the simplicial factorization method.
///
/// Stores symbolic analysis, numeric factorization, row and column permutations.
/// All fields are uninitialized (`None` or empty) until `analyze` and `factorize` are called.
#[allow(non_snake_case)]
pub struct SimplicialSparseLu {
    /// Numeric LU factorization (set by `factorize`).
    lu: Option<SimplicialLu<I, E>>,
    /// Row permutation from pivoting (set by `factorize`).
    row_perm: Option<Perm<I>>,
    /// Column permutation for fill reduction (set by `analyze`).
    col_perm: Option<Perm<I>>,
    /// Matrix dimensions
    nrows: usize,
    ncols: usize,
}

impl Solver for SimplicialSparseLu {
    fn new() -> Self {
        Self {
            lu: None,
            row_perm: None,
            col_perm: None,
            nrows: 0,
            ncols: 0,
        }
    }

    /// Performs symbolic analysis of the input matrix and computes fill-reducing column permutation.
    fn analyze(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), SolverError> {
        let nrows = mat.nrows();
        let ncols = mat.ncols();
        let nnz = mat.compute_nnz();

        self.nrows = nrows;
        self.ncols = ncols;

        // Fill reducing column permutation using COLAMD
        let (col_perm_fwd, col_perm_inv) = {
            let mut perm = Vec::new();
            let mut perm_inv = Vec::new();
            perm.try_reserve_exact(ncols)
                .context(MemoryReservationSnafu {})?;
            perm_inv
                .try_reserve_exact(ncols)
                .context(MemoryReservationSnafu {})?;
            perm.resize(ncols, 0usize);
            perm_inv.resize(ncols, 0usize);

            let mut mem = MemBuffer::try_new(colamd::order_scratch::<usize>(nrows, ncols, nnz))
                .context(MemoryAllocationSnafu {})?;

            colamd::order(
                &mut perm,
                &mut perm_inv,
                mat.symbolic(),
                colamd::Control::default(),
                MemStack::new(&mut mem),
            )
            .context(SymbolicFactorizationSnafu {
                message: "Failed to compute COLAMD ordering".to_string(),
            })?;

            (perm, perm_inv)
        };

        self.col_perm = Some(unsafe {
            Perm::new_unchecked(
                col_perm_fwd.into_boxed_slice(),
                col_perm_inv.into_boxed_slice(),
            )
        });

        Ok(())
    }

    /// Performs numeric LU factorization of the matrix after symbolic analysis.
    fn factorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), SolverError> {
        let col_perm = self.col_perm.as_ref().context(UninitializedSnafu {
            message: "Column permutation",
        })?;

        let nrows = mat.nrows();
        let ncols = mat.ncols();

        // Initialize row permutation
        let mut row_perm = Vec::new();
        let mut row_perm_inv = Vec::new();
        row_perm
            .try_reserve_exact(nrows)
            .context(MemoryReservationSnafu {})?;
        row_perm_inv
            .try_reserve_exact(nrows)
            .context(MemoryReservationSnafu {})?;
        row_perm.resize(nrows, 0usize);
        row_perm_inv.resize(nrows, 0usize);

        // Initialize LU structure
        let mut lu = SimplicialLu::new();

        // Numeric factorization
        let mut mem = MemBuffer::try_new(simplicial::factorize_simplicial_numeric_lu_scratch::<
            I,
            E,
        >(nrows, ncols))
        .context(MemoryAllocationSnafu {})?;
        let mut stack = MemStack::new(&mut mem);

        simplicial::factorize_simplicial_numeric_lu::<I, E>(
            &mut row_perm,
            &mut row_perm_inv,
            &mut lu,
            mat.rb(),
            col_perm.as_ref(),
            &mut stack,
        )
        .context(LuFactorizationSnafu {
            message: "Failed to factorize numeric LU".to_string(),
        })?;

        self.row_perm = Some(unsafe {
            Perm::new_unchecked(row_perm.into_boxed_slice(), row_perm_inv.into_boxed_slice())
        });
        self.lu = Some(lu);

        Ok(())
    }

    /// Refactorizes the matrix.
    fn refactorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), SolverError> {
        self.factorize(mat)
    }

    /// Solves the linear system in place for the given right-hand side vector `b`.
    fn solve_in_place(&self, sol: &mut MatMut<E>) -> Result<(), SolverError> {
        let lu = self.lu.as_ref().context(UninitializedSnafu {
            message: "LU factorization",
        })?;
        let row_perm = self.row_perm.as_ref().context(UninitializedSnafu {
            message: "Row permutation",
        })?;
        let col_perm = self.col_perm.as_ref().context(UninitializedSnafu {
            message: "Column permutation",
        })?;

        let nrows = lu.nrows();
        let nrhs = sol.ncols();

        let mut work = Mat::zeros(nrows, nrhs);

        lu.solve_in_place_with_conj(
            row_perm.as_ref(),
            col_perm.as_ref(),
            faer::Conj::No,
            sol.rb_mut(),
            faer::Par::Seq,
            work.as_mut(),
        );

        Ok(())
    }

    fn solve(&self, b: MatRef<E>) -> Result<Mat<E>, SolverError> {
        let mut sol = Mat::zeros(b.nrows(), b.ncols());
        sol.copy_from(b);
        self.solve_in_place(&mut sol.as_mut())?;
        Ok(sol)
    }
}

impl SimplicialSparseLu {
    /// Creates a new instance of `SimplicialSparseLu` with all fields uninitialized.
    pub fn new() -> Self {
        Self {
            lu: None,
            row_perm: None,
            col_perm: None,
            nrows: 0,
            ncols: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::SparseColMat;

    fn test_lu_solver<T: Solver>(mat: SparseColMat<I, E>, n_count: usize) {
        let mut solver = T::new();
        solver.analyze(mat.as_ref()).unwrap();
        solver.factorize(mat.as_ref()).unwrap();

        use faer::rand::SeedableRng;
        use faer::rand::rngs::StdRng;
        use faer::stats::DistributionExt;
        use faer::stats::prelude::{CwiseMatDistribution, StandardNormal};

        let rng = &mut StdRng::seed_from_u64(0);
        let n = mat.ncols();

        // Generate several random column vectors and verify the results
        for _ in 0..n_count {
            let col = CwiseMatDistribution {
                nrows: n,
                ncols: 1,
                dist: StandardNormal,
            }
            .rand(rng);

            let result = solver.solve(col.as_ref()).expect("Unable to solve");

            println!("SolverError: {:e}", (&col - &mat * &result).norm_l2());
            assert!((&col - &mat * &result).norm_l2() < 1e-10); // Check if Ax ≈ b
        }
    }

    #[test]
    fn test_simplicial_lu_1() {
        // Create a general sparse matrix
        let n = 3;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(faer::sparse::Triplet::new(i, i, 4.0));
            if i + 1 < n {
                triplets.push(faer::sparse::Triplet::new(i, i + 1, -1.0));
                triplets.push(faer::sparse::Triplet::new(i + 1, i, -1.0));
            }
        }
        let mat = faer::sparse::SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();

        test_lu_solver::<SimplicialSparseLu>(mat, 10);
    }
}
