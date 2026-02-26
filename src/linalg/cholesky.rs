//! # Linear Solvers for Symmetric Sparse Matrices
//!
//! This module provides traits and implementations for solving symmetric sparse linear systems,
//! with a focus on Cholesky factorization using the Faer library. It includes:
//! - The [`SymmetricLinearSolver`] trait for a unified solver interface.
//! - The [`SimplicialSparseCholesky`] solver for efficient sparse Cholesky factorization.
//! - Helper functions for matrix permutation and extraction.
//! - Unit tests for correctness and accuracy.
//!
//! ## Example Usage
//! ```
//! use crate::E;
//! use faer::sparse::{SparseColMat, Triplet};
//! use copters::linalg::cholesky::SimplicialSparseCholesky;
//! use copters::linalg::solver::Solver;
//!
//! // Construct a symmetric positive definite sparse matrix
//! let n = 3;
//! let triplets = vec![
//!     Triplet::new(0, 0, 2.0),
//!     Triplet::new(0, 1, -1.0),
//!     Triplet::new(1, 0, -1.0),
//!     Triplet::new(1, 1, 2.0),
//!     Triplet::new(1, 2, -1.0),
//!     Triplet::new(2, 1, -1.0),
//!     Triplet::new(2, 2, 2.0),
//! ];
//! let mat = SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();
//!
//! // Create and use the solver
//! let mut solver = SimplicialSparseCholesky::new();
//! solver.analyze(mat.as_ref()).unwrap();
//! solver.factorize(mat.as_ref()).unwrap();
//! let b = faer::Mat::from_fn(n, 1, |i, _| i as E);
//! let x = solver.solve(b.as_ref()).unwrap();
//! ```
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::ldlt::factor::LdltRegularization;
use faer::perm::{Perm, PermRef};
use faer::prelude::{Reborrow, ReborrowMut};
use faer::sparse::linalg::amd;
use faer::sparse::linalg::cholesky::simplicial::{
    self, SimplicialLdltRef, SymbolicSimplicialCholesky,
};
use faer::sparse::linalg::cholesky::supernodal::{
    self, SupernodalLdltRef, SymbolicSupernodalCholesky,
};
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat};
use faer::{Mat, MatMut, MatRef};
use problemo::{Problem, ProblemResult};

use crate::linalg::solver::{LinearSolver, LinearSolverError, Solver, SymmetricLinearSolver};
use crate::{E, I};

/// Sparse Cholesky solver using the simplicial factorization method.
///
/// Stores symbolic analysis, numeric factorization values, permutation, and LDLT factorization
/// reference. All fields are uninitialized (`None` or empty) until `analyze` and `factorize` are
/// called.
#[allow(non_snake_case)]
pub struct SimplicialSparseCholesky {
    /// Symbolic analysis data for the Cholesky factorization (set by `analyze`).
    symbolic: Option<SymbolicSimplicialCholesky<I>>,
    /// Numeric factorization values (set by `factorize`).
    L_values: Vec<E>,
    /// Permutation used for fill-reducing reordering of the matrix (set by `analyze`).
    perm: Option<Perm<I>>,
    /// LDLT factorization reference (set by `factorize`).
    ldlt: Option<SimplicialLdltRef<'static, I, E>>,
}

/// Implementation of the `SymmetricLinearSolver` trait for the `SimplicialSparseCholesky` solver.
/// Provides symbolic analysis, factorization, and solution routines for symmetric sparse matrices.
impl Solver for SimplicialSparseCholesky {
    fn new() -> Self {
        Self {
            symbolic: None,
            L_values: Vec::new(),
            perm: None,
            ldlt: None,
        }
    }

    /// Performs symbolic analysis of the input matrix, computes fill-reducing permutation,
    /// and prepares internal state for factorization.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn analyze(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem> {
        let nnz = mat.compute_nnz();
        let dim = mat.ncols();
        let mat_symbolic = mat.symbolic();

        // Fill reducing permutation
        let (perm_fwd, perm_inv) = {
            let mut perm = Vec::new();
            let mut perm_inv = Vec::new();
            perm.try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            perm_inv
                .try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            perm.resize(dim, 0usize);
            perm_inv.resize(dim, 0usize);

            let mut mem = MemBuffer::try_new(amd::order_scratch::<I>(dim, nnz))
                .via(LinearSolverError::MemoryAllocation)?;
            amd::order(
                &mut perm,
                &mut perm_inv,
                mat_symbolic,
                amd::Control::default(),
                MemStack::new(&mut mem),
            )
            .via(LinearSolverError::SymbolicFactorization)?;

            (perm, perm_inv)
        };

        self.perm = Some(unsafe {
            Perm::new_unchecked(perm_fwd.into_boxed_slice(), perm_inv.into_boxed_slice())
        });

        let mat_upper = get_mat_upper(mat, self.perm.rb().unwrap().as_ref())?;
        // let mat_upper = self.get_mat_upper(mat);

        // symbolic analysis
        self.symbolic = Some({
            let mut mem = MemBuffer::try_new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<I>(dim, nnz),
                simplicial::factorize_simplicial_symbolic_cholesky_scratch::<I>(dim),
            ]))
            .via(LinearSolverError::MemoryAllocation)?;
            let stack = MemStack::new(&mut mem);

            let mut etree = Vec::new();
            let mut col_counts = Vec::new();
            etree
                .try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            etree.resize(dim, 0isize);
            col_counts
                .try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            col_counts.resize(dim, 0usize);

            simplicial::prefactorize_symbolic_cholesky(
                &mut etree,
                &mut col_counts,
                mat_upper.symbolic(),
                stack,
            );
            simplicial::factorize_simplicial_symbolic_cholesky(
                mat_upper.symbolic(),
                // SAFETY: `etree` was filled correctly by
                // `simplicial::prefactorize_symbolic_cholesky`.
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack,
            )
            .via(LinearSolverError::SymbolicFactorization)?
        });

        // Implementation of analysis
        Ok(())
    }

    /// Performs numeric factorization of the matrix after symbolic analysis.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn factorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem> {
        let symbolic = self
            .symbolic
            .as_ref()
            .ok_or(LinearSolverError::Uninitialized)?;
        let dim = mat.ncols();

        self.L_values = Vec::new();
        self.L_values
            .try_reserve_exact(symbolic.len_val())
            .via(LinearSolverError::MemoryReservation)?;
        self.L_values.resize(symbolic.len_val(), 0.0f64);

        let mat_upper = get_mat_upper(mat, self.perm.rb().unwrap().as_ref())?;
        // let mat_upper = self.get_mat_upper(mat);

        // numerical factorization
        let mut mem = MemBuffer::try_new(StackReq::all_of(&[
            simplicial::factorize_simplicial_numeric_ldlt_scratch::<I, E>(dim),
            // faer::perm::permute_rows_in_place_scratch::<I, E>(dim, 1),
            // symbolic.solve_in_place_scratch::<E>(dim),
        ]))
        .via(LinearSolverError::MemoryAllocation)?;

        let stack = MemStack::new(&mut mem);

        simplicial::factorize_simplicial_numeric_ldlt::<I, E>(
            &mut self.L_values,
            mat_upper.rb(),
            LdltRegularization::default(),
            symbolic,
            stack,
        )
        .via(LinearSolverError::NumericFactorization)?;
        // TODO: consider LdltInfo and LdltErrors

        // SAFETY: We extend the lifetime to 'static because symbolic and L_values are owned by self and
        // live as long as self.
        self.ldlt = Some(unsafe {
            std::mem::transmute::<
                simplicial::SimplicialLdltRef<'_, I, E>,
                simplicial::SimplicialLdltRef<'static, I, E>,
            >(simplicial::SimplicialLdltRef::<'_, I, E>::new(
                symbolic,
                &self.L_values,
            ))
        });

        // Implementation of factorization
        Ok(())
    }

    /// Solves the linear system in place for the given right-hand side vector `b`.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn solve_in_place(&mut self, sol: &mut MatMut<E>) -> Result<(), Problem> {
        let symbolic = self
            .symbolic
            .as_ref()
            .ok_or(LinearSolverError::Uninitialized)?;
        let perm = self.perm.as_ref().ok_or(LinearSolverError::Uninitialized)?;
        let ldlt = self.ldlt.as_ref().ok_or(LinearSolverError::Uninitialized)?;

        let dim = symbolic.ncols();

        let mut mem = MemBuffer::try_new(StackReq::all_of(&[
            // simplicial::factorize_simplicial_numeric_ldlt_scratch::<I, E>(dim),
            faer::perm::permute_rows_in_place_scratch::<I, E>(dim, 1),
            symbolic.solve_in_place_scratch::<E>(dim),
        ]))
        .via(LinearSolverError::MemoryAllocation)?;
        let stack = MemStack::new(&mut mem);

        faer::perm::permute_rows_in_place(sol.rb_mut(), perm.as_ref(), stack);
        ldlt.solve_in_place_with_conj(faer::Conj::No, sol.rb_mut(), faer::Par::Seq, stack);
        faer::perm::permute_rows_in_place(sol.rb_mut(), perm.as_ref().inverse(), stack);

        Ok(())
    }
}

impl LinearSolver for SimplicialSparseCholesky {}

impl SymmetricLinearSolver for SimplicialSparseCholesky {}

impl SimplicialSparseCholesky {
    /// Creates a new instance of `SimplicialSparseCholesky` with all fields uninitialized (set to
    /// `None` or empty).
    ///
    /// # Returns
    ///
    /// A new `SimplicialSparseCholesky` object with:
    /// - `symbolic`: `None` (symbolic analysis not performed)
    /// - `perm`: `None` (permutation not set)
    /// - `L_values`: empty vector (numeric factorization not performed)
    /// - `ldlt`: `None` (LDLT factorization not performed)
    ///
    /// These fields must be properly initialized by calling `analyze` and `factorize` before use.
    pub fn new() -> Self {
        Self {
            symbolic: None,
            perm: None,
            L_values: Vec::new(),
            ldlt: None,
        }
    }
}

/// Sparse Cholesky solver using the simplicial factorization method.
///
/// Stores symbolic analysis, numeric factorization values, permutation, and LDLT factorization
/// reference. All fields are uninitialized (`None` or empty) until `analyze` and `factorize` are
/// called.
#[allow(non_snake_case)]
pub struct SupernodalSparseCholesky {
    /// Symbolic analysis data for the Cholesky factorization (set by `analyze`).
    symbolic: Option<SymbolicSupernodalCholesky<I>>,
    /// Numeric factorization values (set by `factorize`).
    L_values: Vec<E>,
    /// Permutation used for fill-reducing reordering of the matrix (set by `analyze`).
    perm: Option<Perm<I>>,
    /// LDLT factorization reference (set by `factorize`).
    ldlt: Option<SupernodalLdltRef<'static, I, E>>,
}

/// Implementation of the `SymmetricLinearSolver` trait for the `SupernodalSparseCholesky` solver.
/// Provides symbolic analysis, factorization, and solution routines for symmetric sparse matrices.
impl Solver for SupernodalSparseCholesky {
    fn new() -> Self {
        Self {
            symbolic: None,
            L_values: Vec::new(),
            perm: None,
            ldlt: None,
        }
    }

    /// Performs symbolic analysis of the input matrix, computes fill-reducing permutation,
    /// and prepares internal state for factorization.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn analyze(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem> {
        let nnz = mat.compute_nnz();
        let dim = mat.ncols();
        let mat_symbolic = mat.symbolic();

        // Fill reducing permutation
        let (perm_fwd, perm_inv) = {
            let mut perm = Vec::new();
            let mut perm_inv = Vec::new();
            perm.try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            perm_inv
                .try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            perm.resize(dim, 0usize);
            perm_inv.resize(dim, 0usize);

            let mut mem = MemBuffer::try_new(amd::order_scratch::<I>(dim, nnz))
                .via(LinearSolverError::MemoryAllocation)?;
            amd::order(
                &mut perm,
                &mut perm_inv,
                mat_symbolic,
                amd::Control::default(),
                MemStack::new(&mut mem),
            )
            .via(LinearSolverError::SymbolicFactorization)?;

            (perm, perm_inv)
        };

        self.perm = Some(unsafe {
            Perm::new_unchecked(perm_fwd.into_boxed_slice(), perm_inv.into_boxed_slice())
        });

        // let mat_upper = self.get_mat_upper(mat);
        let mat_upper = get_mat_upper(mat, self.perm.rb().unwrap().as_ref())?;

        // symbolic analysis
        self.symbolic = Some({
            let mut mem = MemBuffer::try_new(StackReq::any_of(&[
                simplicial::prefactorize_symbolic_cholesky_scratch::<I>(dim, nnz),
                supernodal::factorize_supernodal_symbolic_cholesky_scratch::<I>(dim),
            ]))
            .via(LinearSolverError::MemoryAllocation)?;
            let stack = MemStack::new(&mut mem);

            let mut etree = Vec::new();
            let mut col_counts = Vec::new();
            etree
                .try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            etree.resize(dim, 0isize);
            col_counts
                .try_reserve_exact(dim)
                .via(LinearSolverError::MemoryReservation)?;
            col_counts.resize(dim, 0usize);

            simplicial::prefactorize_symbolic_cholesky(
                &mut etree,
                &mut col_counts,
                mat_upper.symbolic(),
                stack,
            );
            supernodal::factorize_supernodal_symbolic_cholesky(
                mat_upper.symbolic(),
                // SAFETY: `etree` was filled correctly by
                // `simplicial::prefactorize_symbolic_cholesky`.
                unsafe { simplicial::EliminationTreeRef::from_inner(&etree) },
                &col_counts,
                stack,
                faer::sparse::linalg::SymbolicSupernodalParams { relax: None },
            )
            .via(LinearSolverError::SymbolicFactorization)?
        });

        // Implementation of analysis
        Ok(())
    }

    /// Performs numeric factorization of the matrix after symbolic analysis.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn factorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem> {
        let symbolic = self
            .symbolic
            .as_ref()
            .ok_or(LinearSolverError::Uninitialized)?;
        let _dim = mat.ncols();

        self.L_values = Vec::new();
        self.L_values
            .try_reserve_exact(symbolic.len_val())
            .via(LinearSolverError::MemoryReservation)?;
        self.L_values.resize(symbolic.len_val(), 0.0f64);

        let mat_lower = get_mat_lower(mat, self.perm.rb().unwrap().as_ref())?;
        // let mat_lower = self.get_mat_lower(mat);

        // numerical factorization
        let mut mem = MemBuffer::try_new(StackReq::all_of(&[
            supernodal::factorize_supernodal_numeric_ldlt_scratch::<I, E>(
                symbolic,
                faer::Par::Seq,
                Default::default(),
            ),
            // faer::perm::permute_rows_in_place_scratch::<I, E>(dim, 1),
            // symbolic.solve_in_place_scratch::<E>(dim, faer::Par::Seq),
        ]))
        .via(LinearSolverError::MemoryAllocation)?;

        let stack = MemStack::new(&mut mem);

        supernodal::factorize_supernodal_numeric_ldlt::<I, E>(
            &mut self.L_values,
            mat_lower.rb(),
            LdltRegularization::default(),
            symbolic,
            faer::Par::Seq,
            stack,
            Default::default(),
        )
        .via(LinearSolverError::NumericFactorization)?;
        // TODO: consider LdltInfo and LdltErrors

        // SAFETY: We extend the lifetime to 'static because symbolic and L_values are owned by self and
        // live as long as self.
        self.ldlt = Some(unsafe {
            std::mem::transmute::<
                supernodal::SupernodalLdltRef<'_, I, E>,
                supernodal::SupernodalLdltRef<'static, I, E>,
            >(supernodal::SupernodalLdltRef::<'_, I, E>::new(
                symbolic,
                &self.L_values,
            ))
        });

        // Implementation of factorization
        Ok(())
    }

    /// Refactorizes the matrix, typically used when the matrix structure remains but values change.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn refactorize(&mut self, mat: SparseColMatRef<I, E>) -> Result<(), Problem> {
        // Implementation of refactorization
        self.factorize(mat)
    }

    /// Solves the linear system in place for the given right-hand side vector `b`.
    /// Returns `Ok(())` on success, or an error message on failure.
    fn solve_in_place(&mut self, sol: &mut MatMut<E>) -> Result<(), Problem> {
        let symbolic = self
            .symbolic
            .as_ref()
            .ok_or(LinearSolverError::Uninitialized)?;
        let perm = self.perm.as_ref().ok_or(LinearSolverError::Uninitialized)?;
        let ldlt = self.ldlt.as_ref().ok_or(LinearSolverError::Uninitialized)?;

        let dim = symbolic.ncols();

        let mut mem = MemBuffer::try_new(StackReq::all_of(&[
            // supernodal::factorize_supernodal_numeric_ldlt_scratch::<I, E>(
            //     symbolic,
            //     faer::Par::Seq,
            //     Default::default(),
            // ),
            faer::perm::permute_rows_in_place_scratch::<I, E>(dim, 1),
            symbolic.solve_in_place_scratch::<E>(dim, faer::Par::Seq),
        ]))
        .via(LinearSolverError::MemoryAllocation)?;
        let stack = MemStack::new(&mut mem);

        faer::perm::permute_rows_in_place(sol.rb_mut(), perm.as_ref(), stack);
        ldlt.solve_in_place_with_conj(faer::Conj::No, sol.rb_mut(), faer::Par::Seq, stack);
        faer::perm::permute_rows_in_place(sol.rb_mut(), perm.as_ref().inverse(), stack);

        Ok(())
    }
}

impl LinearSolver for SupernodalSparseCholesky {}

impl SymmetricLinearSolver for SupernodalSparseCholesky {}

impl SupernodalSparseCholesky {
    /// Creates a new instance of `SupernodalSparseCholesky` with all fields uninitialized (set to
    /// `None` or empty).
    ///
    /// # Returns
    ///
    /// A new `SupernodalSparseCholesky` object with:
    /// - `symbolic`: `None` (symbolic analysis not performed)
    /// - `perm`: `None` (permutation not set)
    /// - `L_values`: empty vector (numeric factorization not performed)
    /// - `ldlt`: `None` (LDLT factorization not performed)
    ///
    /// These fields must be properly initialized by calling `analyze` and `factorize` before use.
    pub fn new() -> Self {
        Self {
            symbolic: None,
            perm: None,
            L_values: Vec::new(),
            ldlt: None,
        }
    }
}

fn get_mat_lower(
    mat: SparseColMatRef<I, E>,
    perm: PermRef<I>,
) -> Result<SparseColMat<I, E>, Problem> {
    let dim = mat.ncols();
    let nnz = mat.compute_nnz();

    let mut mat_col_ptrs = Vec::new();
    let mut mat_row_indices = Vec::new();
    let mut mat_values = Vec::new();

    mat_col_ptrs
        .try_reserve_exact(dim + 1)
        .via(LinearSolverError::MemoryReservation)?;
    mat_col_ptrs.resize(dim + 1, 0usize);
    mat_row_indices
        .try_reserve_exact(nnz)
        .via(LinearSolverError::MemoryReservation)?;
    mat_row_indices.resize(nnz, 0usize);
    mat_values
        .try_reserve_exact(nnz)
        .via(LinearSolverError::MemoryReservation)?;
    mat_values.resize(nnz, 0.0f64);

    let mut mem = MemBuffer::try_new(faer::sparse::utils::permute_self_adjoint_scratch::<I>(dim))
        .via(LinearSolverError::MemoryAllocation)?;
    faer::sparse::utils::permute_self_adjoint_to_unsorted(
        &mut mat_values,
        &mut mat_col_ptrs,
        &mut mat_row_indices,
        mat.rb(),
        perm.rb(),
        faer::Side::Lower,
        faer::Side::Lower,
        MemStack::new(&mut mem),
    );

    Ok(SparseColMat::<I, E>::new(
        unsafe {
            SymbolicSparseColMat::new_unchecked(dim, dim, mat_col_ptrs, None, mat_row_indices)
        },
        mat_values,
    ))
}

fn get_mat_upper(
    mat: SparseColMatRef<I, E>,
    perm: PermRef<I>,
) -> Result<SparseColMat<I, E>, Problem> {
    let dim = mat.ncols();
    let nnz = mat.compute_nnz();

    let mut mat_col_ptrs = Vec::new();
    let mut mat_row_indices = Vec::new();
    let mut mat_values = Vec::new();

    mat_col_ptrs
        .try_reserve_exact(dim + 1)
        .via(LinearSolverError::MemoryReservation)?;
    mat_col_ptrs.resize(dim + 1, 0usize);
    mat_row_indices
        .try_reserve_exact(nnz)
        .via(LinearSolverError::MemoryReservation)?;
    mat_row_indices.resize(nnz, 0usize);
    mat_values
        .try_reserve_exact(nnz)
        .via(LinearSolverError::MemoryReservation)?;
    mat_values.resize(nnz, 0.0f64);

    let mut mem = MemBuffer::try_new(faer::sparse::utils::permute_self_adjoint_scratch::<I>(dim))
        .via(LinearSolverError::MemoryAllocation)?;
    faer::sparse::utils::permute_self_adjoint_to_unsorted(
        &mut mat_values,
        &mut mat_col_ptrs,
        &mut mat_row_indices,
        mat.rb(),
        perm.rb(),
        faer::Side::Upper,
        faer::Side::Upper,
        MemStack::new(&mut mem),
    );

    Ok(SparseColMat::<I, E>::new(
        unsafe {
            SymbolicSparseColMat::new_unchecked(dim, dim, mat_col_ptrs, None, mat_row_indices)
        },
        mat_values,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::rand::SeedableRng;
    use faer::rand::rngs::StdRng;
    use faer::stats::DistributionExt;
    use faer::stats::prelude::{CwiseMatDistribution, StandardNormal};
    use rstest::rstest;
    use rstest_reuse::{apply, template};

    enum SolverType {
        SimplicialCholesky,
        SupernodalCholesky,
    }

    fn test_symmetric_solver(mat: SparseColMat<I, E>, solver_type: SolverType, n_count: I) {
        let mut solver: Box<dyn SymmetricLinearSolver> = match solver_type {
            SolverType::SimplicialCholesky => Box::new(SimplicialSparseCholesky::new()),
            SolverType::SupernodalCholesky => Box::new(SupernodalSparseCholesky::new()),
        };
        solver.analyze(mat.as_ref()).unwrap();
        solver.factorize(mat.as_ref()).unwrap();

        let rng = &mut StdRng::seed_from_u64(0);

        let n = mat.ncols();

        // Generate several random column vectors and verify whether the results hold
        for _ in 0..n_count {
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

    #[template]
    #[rstest]
    fn test_symmetric_solver_1(
        #[values(SolverType::SimplicialCholesky, SolverType::SupernodalCholesky)]
        solver_type: SolverType,
    ) {
        // Create the SPD matrix as triplets
        let n = 3;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push(faer::sparse::Triplet::new(i, i, 2.0));
            if i + 1 < n {
                triplets.push(faer::sparse::Triplet::new(i, i + 1, -1.0));
                triplets.push(faer::sparse::Triplet::new(i + 1, i, -1.0));
            }
        }
        let mat = faer::sparse::SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();

        test_symmetric_solver(mat, solver_type, 10);
    }

    #[apply(test_symmetric_solver_1)]
    fn test_symmetric_solver_trefethan20b(solver_type: SolverType) {
        let mat = loaders::mtx::get_matrix_by_name("Trefethen 20b", true);
        test_symmetric_solver(mat, solver_type, 10);
    }
}
