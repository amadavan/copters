use std::marker::PhantomData;

use faer::{
    col::generic::Col,
    prelude::ReborrowMut,
    sparse::{SparseColMat, SymbolicSparseColMat},
};
use problemo::Problem;

use crate::{
    E, I, Residual, SolverState,
    linalg::{
        solver::LinearSolver,
        vector_ops::{cwise_inverse, cwise_multiply},
    },
    qp::{QuadraticProgram, mpc::Step},
};

/// Formulation and factorization of the augmented KKT system used to
/// compute search directions in a primal-dual interior-point method.
pub trait AugmentedSystem<'a, Solver: LinearSolver> {
    /// Creates a new instance, performing symbolic analysis of the sparsity pattern.
    fn new(qp: &'a QuadraticProgram) -> Self
    where
        Self: Sized;

    /// Updates the numeric values, re-factorizes, and solves for a search direction.
    fn solve(&mut self, state: &SolverState, rhs: &Residual) -> Result<Step, Problem>;

    /// Solves for a search direction reusing the current factorization.
    fn resolve(&mut self, state: &SolverState, rhs: &Residual) -> Result<Step, Problem>;
}

/// Standard augmented system formulation.
///
/// Assembles and solves the `(n_var + n_con) x (n_var + n_con)` system:
///
/// ```text
/// [ Q-D   A^T ] [ dx ] = [ r_d + z_l + z_u - sigma*mu*(X-L)^{-1}e - sigma*mu*(X-U)^{-1}e ]
/// [  A    0  ] [ dy ]   [ r_p                                                              ]
/// ```
///
/// where `D = Z_l (X-L)^{-1} + Z_u (X-U)^{-1}`. The dual directions
/// `dz_l` and `dz_u` are recovered from `dx` after the solve.
pub struct StandardSystem<'a, Solver: LinearSolver> {
    qp: &'a QuadraticProgram,
    mat: SparseColMat<I, E>,
    solver: Solver,
    diag_idx: Vec<I>, // Indices of the diagonal entries corresponding to dx in the matrix

    _a: PhantomData<&'a ()>,
}

impl<'a, Solver: LinearSolver> AugmentedSystem<'a, Solver> for StandardSystem<'a, Solver> {
    fn new(qp: &'a QuadraticProgram) -> Self {
        // Get properties
        let (n_var, n_con) = qp.get_dims();
        let a_nnz = qp.A.compute_nnz();
        let q_nnz = qp.Q.compute_nnz();
        let n_values = n_var + 2 * a_nnz + q_nnz;

        let mut col_ptrs = Vec::with_capacity(n_var + n_con + 1);
        let mut row_indices = Vec::with_capacity(n_values);
        let mut values = Vec::with_capacity(n_values);

        // Set pointers and values for the first n_var columns (dx)
        let q_col_ptr = qp.Q.symbolic().col_ptr();
        let q_row_idx = qp.Q.symbolic().row_idx();
        let q_values = qp.Q.val();

        let a_col_ptr = qp.A.symbolic().col_ptr();
        let a_row_idx = qp.A.symbolic().row_idx();
        let a_values = qp.A.val();

        // Set each column (0...n_var)
        // TODO: ensure diagonals exist for dx and are set to -1, then only store the off-diagonal values of Q
        let mut diag_idx = Vec::with_capacity(n_var);
        col_ptrs.push(0);
        for j in 0..n_var {
            let mut has_diag = false;
            if j < q_col_ptr.len() {
                let start = q_col_ptr[j];
                let end = q_col_ptr[j + 1];
                for k in start..end {
                    if k == j {
                        // Add the diagonal contribution from the complementarity terms
                        row_indices.push(q_row_idx[k]); // Hessian part for dx
                        values.push(q_values[k] + 1.); // Identity part for dx
                        diag_idx.push(row_indices.len() - 1); // Store index of diagonal for later updates
                        has_diag = true;
                    } else if k != end - 1 && j > q_row_idx[k] && j < q_row_idx[k + 1] {
                        // If the diagonal was skipped make sure to add it
                        row_indices.push(j); // Diagonal part for dx
                        values.push(1.);
                        diag_idx.push(row_indices.len() - 1); // Store index of diagonal for later updates
                        has_diag = true;

                        row_indices.push(q_row_idx[k]); // Hessian part for dx
                        values.push(q_values[k]);
                    } else {
                        // Just add it normally
                        row_indices.push(q_row_idx[k]); // Hessian part for dx
                        values.push(q_values[k]);
                    }
                }
            }

            // Add diagonal if it was not present in the Hessian (i.e. last element was before the diagonal)
            if !has_diag {
                row_indices.push(j); // Diagonal part for dx
                values.push(1.);
                diag_idx.push(row_indices.len() - 1); // Store index of diagonal for later updates
            }

            let start = a_col_ptr[j];
            let end = a_col_ptr[j + 1];
            for k in start..end {
                row_indices.push(a_row_idx[k] + n_var); // A part for dx
                values.push(-a_values[k]);
            }

            col_ptrs.push(row_indices.len());
        }

        // Set pointers for A^T
        let a_csr = qp.A.to_row_major().unwrap();
        let a_row_ptr = a_csr.symbolic().row_ptr();
        let a_col_idx = a_csr.symbolic().col_idx();
        let a_values = a_csr.val();

        // Set columns for A^T
        for j in 0..n_con {
            let start = a_row_ptr[j];
            let end = a_row_ptr[j + 1];
            for k in start..end {
                row_indices.push(a_col_idx[k]); // A^T part for dy
                values.push(-a_values[k]);
            }

            col_ptrs.push(row_indices.len());
        }

        let mat = unsafe {
            let sym = SymbolicSparseColMat::new_unchecked(
                n_var + n_con,
                n_var + n_con,
                col_ptrs,
                None,
                row_indices,
            );
            SparseColMat::<I, E>::new(sym, values)
        };

        let mut solver = Solver::new();
        solver.analyze(mat.as_ref()).unwrap();

        Self {
            qp,
            mat,
            solver,
            diag_idx,

            _a: PhantomData,
        }
    }

    fn solve(&mut self, state: &SolverState, rhs: &Residual) -> Result<Step, Problem> {
        // Get necessary values
        let xl_inv = cwise_inverse((&state.x - &self.qp.l).as_ref());
        let xu_inv = cwise_inverse((&state.x - &self.qp.u).as_ref());
        let sys_diag = cwise_multiply(xl_inv.as_ref(), state.z_l.as_ref())
            + cwise_multiply(xu_inv.as_ref(), state.z_u.as_ref());

        // Get matrix pointers
        let mat = self.mat.rb_mut();
        let _col_ptrs = mat.symbolic().col_ptr();
        let values = mat.val_mut();

        // Update the matrix values based on the current iterate
        for j in 0..self.qp.get_n_vars() {
            let val = self.qp.Q.get(j, j).unwrap_or(&0.0);
            values[self.diag_idx[j]] = val + sys_diag[j] as E; // Identity part for dx
        }

        self.solver.factorize(self.mat.as_ref())?;

        self.resolve(state, rhs)
    }

    fn resolve(&mut self, state: &SolverState, residual: &Residual) -> Result<Step, Problem> {
        let (n_var, n_con) = self.qp.get_dims();

        // Convert residual to right hand side for the linear system
        let (sigma, mu) = (state.sigma.unwrap(), state.mu.unwrap());
        let mut rhs = Col::zeros(n_var + n_con);
        let xl_inv = cwise_inverse((&state.x - &self.qp.l).as_ref());
        let xu_inv = cwise_inverse((&state.x - &self.qp.u).as_ref());

        let (mut rhs_dual, mut rhs_primal) = rhs.split_at_row_mut(n_var);
        rhs_dual.copy_from(
            residual.get_dual_feasibility()
                + cwise_multiply(xl_inv.as_ref(), residual.cs_lower.as_ref())
                + cwise_multiply(xu_inv.as_ref(), residual.cs_upper.as_ref())
                + sigma * mu * (&xl_inv + &xu_inv),
        );
        rhs_primal.copy_from(&residual.get_primal_feasibility().as_ref());

        let solution = {
            let sol = self.solver.solve(rhs.as_mat().as_ref())?;
            sol.col(0).to_owned()
        };
        let (dx, dy) = solution.split_at_row(n_var);
        let dz_l = sigma * mu * xl_inv.as_ref()
            - cwise_multiply(
                cwise_multiply(xl_inv.as_ref(), state.z_l.as_ref()).as_ref(),
                dx.as_ref(),
            )
            + cwise_multiply(xl_inv.as_ref(), residual.cs_lower.as_ref());
        let dz_u = sigma * mu * xu_inv.as_ref()
            - cwise_multiply(
                cwise_multiply(xu_inv.as_ref(), state.z_u.as_ref()).as_ref(),
                dx.as_ref(),
            )
            + cwise_multiply(xu_inv.as_ref(), residual.cs_upper.as_ref());

        Ok(Step {
            dx: dx.to_owned(), // Placeholder
            dy: dy.to_owned(), // Placeholder
            dz_l,              // Placeholder
            dz_u,              // Placeholder
        })
    }
}
