use std::marker::PhantomData;

use faer::{
    col::generic::Col,
    prelude::ReborrowMut,
    sparse::{SparseColMat, SymbolicSparseColMat},
};
use problemo::Problem;

use crate::{
    E, I, SolverState,
    linalg::{
        solver::LinearSolver,
        vector_ops::{cwise_inverse, cwise_multiply},
    },
    lp::{
        LinearProgram,
        mpc::{Residual, Step},
    },
};

/// Formulation and factorization of the augmented KKT system used to
/// compute search directions in a primal-dual interior-point method.
pub trait AugmentedSystem<'a, Solver: LinearSolver> {
    /// Creates a new instance, performing symbolic analysis of the sparsity pattern.
    fn new(lp: &'a LinearProgram) -> Self
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
/// [ -D   A^T ] [ dx ] = [ r_d + z_l + z_u - sigma*mu*(X-L)^{-1}e - sigma*mu*(X-U)^{-1}e ]
/// [  A    0  ] [ dy ]   [ r_p                                                              ]
/// ```
///
/// where `D = Z_l (X-L)^{-1} + Z_u (X-U)^{-1}`. The dual directions
/// `dz_l` and `dz_u` are recovered from `dx` after the solve.
pub struct StandardSystem<'a, Solver: LinearSolver> {
    lp: &'a LinearProgram,
    mat: SparseColMat<I, E>,
    solver: Solver,

    _a: PhantomData<&'a ()>,
}

impl<'a, Solver: LinearSolver> AugmentedSystem<'a, Solver> for StandardSystem<'a, Solver> {
    fn new(lp: &'a LinearProgram) -> Self {
        // Get properties
        let (n_var, n_con) = lp.get_dims();
        let a_nnz = lp.A.compute_nnz();
        let n_values = n_var + 2 * a_nnz;

        let mut col_ptrs = Vec::with_capacity(n_var + n_con + 1);
        let mut row_indices = Vec::with_capacity(n_values);
        let mut values = Vec::with_capacity(n_values);

        // Set pointers for the first n_var columns (dx)
        let a_col_ptr = lp.A.symbolic().col_ptr();
        let a_row_idx = lp.A.symbolic().row_idx();
        let a_values = lp.A.val();

        // Set each column (0...n_var)
        col_ptrs.push(0);
        for j in 0..n_var {
            row_indices.push(j); // Identity part for dx
            values.push(E::from(1.));

            let start = a_col_ptr[j];
            let end = a_col_ptr[j + 1];
            for k in start..end {
                row_indices.push(a_row_idx[k] + n_var); // A part for dx
                values.push(a_values[k]);
            }

            col_ptrs.push(row_indices.len());
        }

        // Set pointers for A^T
        let a_csr = lp.A.to_row_major().unwrap();
        let a_row_ptr = a_csr.symbolic().row_ptr();
        let a_col_idx = a_csr.symbolic().col_idx();
        let a_values = a_csr.val();

        // Set columns for A^T
        for j in 0..n_con {
            let start = a_row_ptr[j];
            let end = a_row_ptr[j + 1];
            for k in start..end {
                row_indices.push(a_col_idx[k]); // A^T part for dy
                values.push(a_values[k]);
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
            lp,
            mat,
            solver,

            _a: PhantomData,
        }
    }

    fn solve(&mut self, state: &SolverState, rhs: &Residual) -> Result<Step, Problem> {
        // Get necessary values
        let xl_inv = cwise_inverse((&state.x - &self.lp.l).as_ref());
        let xu_inv = cwise_inverse((&state.x - &self.lp.u).as_ref());
        let sys_diag = cwise_multiply(xl_inv.as_ref(), state.z_l.as_ref())
            + cwise_multiply(xu_inv.as_ref(), state.z_u.as_ref());

        // Get matrix pointers
        let mat = self.mat.rb_mut();
        let col_ptrs = mat.symbolic().col_ptr();
        let values = mat.val_mut();

        // Update the matrix
        for j in 0..self.lp.get_n_vars() {
            values[col_ptrs[j]] = -sys_diag[j] as E; // Identity part for dx
        }

        self.solver.factorize(self.mat.as_ref())?;

        self.resolve(state, rhs)
    }

    fn resolve(&mut self, state: &SolverState, residual: &Residual) -> Result<Step, Problem> {
        let (n_var, n_con) = self.lp.get_dims();

        // Convert residual to right hand side for the linear system
        let (sigma, mu) = state.get_sigma_mu();
        let mut rhs = Col::zeros(n_var + n_con);
        let xl_inv = cwise_inverse((&state.x - &self.lp.l).as_ref());
        let xu_inv = cwise_inverse((&state.x - &self.lp.u).as_ref());

        let (mut rhs_dual, mut rhs_primal) = rhs.split_at_row_mut(n_var);
        rhs_dual.copy_from(
            residual.get_dual_feasibility() + state.z_l.as_ref() + state.z_u.as_ref()
                - sigma * mu * (&xl_inv + &xu_inv),
        );
        rhs_primal.copy_from(&residual.get_primal_feasibility().as_ref());

        let solution = {
            let sol = self.solver.solve(rhs.as_mat().as_ref())?;
            sol.col(0).to_owned()
        };
        let (dx, dy) = solution.split_at_row(n_var);
        let dz_l = sigma * mu * xl_inv.as_ref()
            - &state.z_l
            - cwise_multiply(
                cwise_multiply(xl_inv.as_ref(), state.z_l.as_ref()).as_ref(),
                dx.as_ref(),
            );
        let dz_u = sigma * mu * xu_inv.as_ref()
            - &state.z_u
            - cwise_multiply(
                cwise_multiply(xu_inv.as_ref(), state.z_u.as_ref()).as_ref(),
                dx.as_ref(),
            );

        Ok(Step {
            dx: dx.to_owned(), // Placeholder
            dy: dy.to_owned(), // Placeholder
            dz_l,              // Placeholder
            dz_u,              // Placeholder
        })
    }
}
