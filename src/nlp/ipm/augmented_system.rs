use faer::{
    Col,
    sparse::{SparseColMat, SymbolicSparseColMat},
};
use macros::explicit_options;

use crate::{
    E, I, Residual, SolverOptions, SolverState,
    linalg::{
        solver::LinearSolver,
        vector_ops::{cwise_inverse, cwise_multiply},
    },
    nlp::{NonlinearProgram, ipm::Step},
};

pub trait AugmentedSystem<'a, LinSolve: LinearSolver> {
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self;

    /// Solves the augmented system for the given right-hand side and returns the solution.
    ///
    /// # Arguments
    ///
    /// * `state` - The current state of the solver, containing the current iterate and other relevant information.
    /// * `rhs` - The right-hand side vector of the augmented system.
    ///
    /// # Returns
    ///
    /// The solution vector of the augmented system.
    fn solve(&mut self, state: &SolverState, rhs: &Residual) -> Step;

    fn resolve(&mut self, state: &SolverState, rhs: &Residual) -> Step;
}

#[explicit_options(name = SolverOptions)]
pub struct StandardSystem<'a, LinSolve: LinearSolver> {
    nlp: &'a NonlinearProgram,
    mat: SparseColMat<I, E>,
    solver: LinSolve,
}

impl<'a, LinSolve: LinearSolver> StandardSystem<'a, LinSolve> {
    fn assemble_matrix(&mut self, state: &SolverState, rhs: &Residual) -> SparseColMat<I, E> {
        // Assemble the augmented system matrix based on the current state and the problem data
        // This typically involves computing the Hessian of the Lagrangian and the Jacobian of the constraints
        let xl_inv = {
            if let Some(l) = &self.nlp.l {
                cwise_inverse((&state.x - l).as_ref())
            } else {
                cwise_inverse(state.x.as_ref())
            }
        };
        let xu_inv = {
            if let Some(u) = &self.nlp.u {
                cwise_inverse((&state.x - u).as_ref())
            } else {
                Col::<E>::zeros(state.x.nrows())
            }
        };
        let sys_diag = cwise_multiply(xl_inv.as_ref(), state.z_l.as_ref())
            + cwise_multiply(xu_inv.as_ref(), state.z_u.as_ref());

        // Update the first n_var columns
        let h_zero =
            SparseColMat::<I, E>::try_new_from_triplets(self.nlp.n_var, self.nlp.n_var, &[])
                .unwrap();
        let h = if let Some(h) = state.h.as_ref() {
            h
        } else {
            &h_zero
        };

        let dg = (&self.nlp.dg)(&state.x);

        let mat_nnz = h.compute_nnz() + dg.compute_nnz() * 2 + self.nlp.n_var; // Hessian + Jacobian + diagonal

        let mut col_ptrs = Vec::with_capacity(self.nlp.n_var + self.nlp.n_cons + 1);
        let mut values = Vec::with_capacity(mat_nnz);
        let mut row_indices = Vec::with_capacity(mat_nnz);

        let h_col_ptr = h.symbolic().col_ptr();
        let h_row_idx = h.symbolic().row_idx();
        let h_vals = h.val();

        let dg_col_ptr = dg.symbolic().col_ptr();
        let dg_row_idx = dg.symbolic().row_idx();
        let dg_vals = dg.val();

        col_ptrs.push(0);
        for j in 0..self.nlp.n_var {
            let start = h_col_ptr[j];
            let end = h_col_ptr[j + 1];
            let mut has_diag = false;
            for k in start..end {
                if k == j {
                    // Add the diagonal contribution from the complementarity terms
                    row_indices.push(h_row_idx[k]); // Hessian part for dx
                    values.push(h_vals[k] + sys_diag[j]);
                    has_diag = true;
                } else if k != end - 1 && j > h_row_idx[k] && j < h_row_idx[k + 1] {
                    // If the diagonal was skipped make sure to add it
                    row_indices.push(j); // Diagonal part for dx
                    values.push(sys_diag[j]);
                    has_diag = true;
                    row_indices.push(h_row_idx[k]); // Hessian part for dx
                    values.push(h_vals[k]);
                } else {
                    // Just add it normally
                    row_indices.push(h_row_idx[k]); // Hessian part for dx
                    values.push(h_vals[k]);
                }
            }

            // Add diagonal if it was not present in the Hessian (i.e. last element was before the diagonal)
            if !has_diag {
                row_indices.push(j); // Diagonal part for dx
                values.push(sys_diag[j]);
            }

            let start = dg_col_ptr[j];
            let end = dg_col_ptr[j + 1];
            for k in start..end {
                row_indices.push(dg_row_idx[k]); // Jacobian part for dy
                values.push(dg_vals[k]);
            }
        }

        let dg_csr = dg.to_row_major().unwrap();
        let dg_row_ptr = dg_csr.symbolic().row_ptr();
        let dg_col_idx = dg_csr.symbolic().col_idx();
        let dg_vals = dg_csr.val();

        // Set pointers for A^T
        for j in 0..self.nlp.n_cons {
            let start = dg_row_ptr[j];
            let end = dg_row_ptr[j + 1];
            for k in start..end {
                row_indices.push(dg_col_idx[k]); // Jacobian part for dy
                values.push(dg_vals[k]);
            }

            col_ptrs.push(row_indices.len());
        }

        let mat = unsafe {
            let sym = SymbolicSparseColMat::<I>::new_unchecked(
                self.nlp.n_var + self.nlp.n_cons,
                self.nlp.n_var + self.nlp.n_cons,
                col_ptrs,
                None,
                row_indices,
            );
            SparseColMat::<I, E>::new(sym, values)
        };

        // TODO: consider that df/dg may be expensive and should only be computed once

        mat
    }
}

impl<'a, LinSolve: LinearSolver> AugmentedSystem<'a, LinSolve> for StandardSystem<'a, LinSolve> {
    fn new(nlp: &'a NonlinearProgram, options: &SolverOptions) -> Self {
        Self {
            nlp,
            mat: SparseColMat::<I, E>::try_new_from_triplets(nlp.n_var, nlp.n_cons, &[]).unwrap(),
            solver: LinSolve::new(),
            options: options.into(),
        }
    }

    fn solve(&mut self, state: &SolverState, rhs: &Residual) -> Step {
        // Matrix may change at each iteration, due to nonlinearity. The entire system matrix must be reconstructed each time.
        let mat = self.assemble_matrix(state, rhs);

        // Check if the matrix has changed sufficiently
        // In a real implementation, we would want to be more sophisticated about when to refactor the matrix
        // For example, we could check if the structure has changed or if the values have changed significantly
        // For now, we will just refactor every time for simplicity
        let structural_change = {
            let current_sym = self.mat.symbolic();
            let new_sym = mat.symbolic();

            current_sym.col_ptr() != new_sym.col_ptr() || current_sym.row_idx() != new_sym.row_idx()
        };

        let value_change = {
            let current_vals = self.mat.val();
            let new_vals = mat.val();

            current_vals.len() != new_vals.len()
                || current_vals
                    .iter()
                    .zip(new_vals.iter())
                    .any(|(a, b)| (*a - *b).abs() > E::from(1e-8))
        };

        if structural_change {
            self.solver.analyze(mat.as_ref()).unwrap();
        }

        if structural_change || value_change {
            self.solver.factorize(mat.as_ref()).unwrap();
            self.mat = mat;
        }

        self.resolve(state, rhs)
    }

    fn resolve(&mut self, state: &SolverState, rhs: &Residual) -> Step {
        todo!()
    }
}
