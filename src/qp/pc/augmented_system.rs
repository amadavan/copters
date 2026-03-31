use faer::{
    Col,
    sparse::{SparseColMat, SymbolicSparseColMat},
    unzip, zip,
};

use crate::{
    E, I, SolverOptions,
    linalg::solver::LinearSolver,
    qp::{QuadraticProgram, pc::Workspace},
    state::{Delta, SolverState, View},
};

type P = QuadraticProgram;

pub trait AugmentedSystem: Sized {
    fn new(options: &SolverOptions) -> Self
    where
        Self: Sized;

    fn analyze_sys(&mut self, view: &View<P, Workspace>);

    fn solve(&mut self, view: &View<P, Workspace>) -> Delta;

    fn factorize_sys(&mut self, view: &View<P, Workspace>);
}

pub struct KKTSystem<Solver: LinearSolver> {
    mat: SparseColMat<I, E>,
    solver: Solver,
    diag_val: Col<E>, // Diagonal entries of Q, used for efficient updates in the augmented system
    diag_idx: Vec<I>, // Indices of the diagonal entries in the matrix
                      // Additional fields for factorization, preconditioners, etc. can be added here
}

impl<Solver: LinearSolver> AugmentedSystem for KKTSystem<Solver> {
    fn new(_options: &SolverOptions) -> Self {
        // Placeholder for initialization, actual matrix construction will happen in analyze_sys
        Self {
            mat: SparseColMat::try_new_from_nonnegative_triplets(0, 0, &[]).unwrap(),
            solver: Solver::new(),
            diag_val: Col::zeros(0),
            diag_idx: Vec::new(),
        }
    }

    fn analyze_sys(&mut self, view: &View<P, Workspace>) {
        let View {
            program: qp,
            state,
            work: _,
        } = view;

        // Construct the KKT matrix based on the current state and qp
        let n_var = state.variables().x().nrows();
        let n_con = state.variables().y().nrows();

        let a_nnz = qp.A.compute_nnz();
        let q_nnz = qp.Q.compute_nnz();
        let n_values = n_var + 2 * a_nnz + q_nnz;

        let q_col_ptrs = qp.Q.col_ptr();
        let q_row_idx = qp.Q.row_idx();
        let q_values = qp.Q.val();

        let a_col_ptrs = qp.A.col_ptr();
        let a_row_idx = qp.A.row_idx();
        let a_values = qp.A.val();

        let mut col_ptrs = Vec::with_capacity(n_var + n_con + 1);
        let mut row_idx = Vec::with_capacity(n_values);
        let mut values = Vec::with_capacity(n_values);

        let mut diag_val = Col::zeros(n_var);
        let mut diag_idx = Vec::with_capacity(n_var);

        // Create [ Q + I ]
        //        [   A   ]
        for j in 0..n_var {
            col_ptrs.push(row_idx.len() as I);
            let mut has_q_diag = false;
            for idx in q_col_ptrs[j]..q_col_ptrs[j + 1] {
                // Q skipped over the diagonal
                if !has_q_diag && q_row_idx[idx] > j {
                    diag_val[j] = 0.0;
                    row_idx.push(j);
                    values.push(1.0);
                }

                let i = q_row_idx[idx];
                let val = q_values[idx];
                row_idx.push(i);
                values.push(val);

                // Q has a diagonal
                if i == j {
                    diag_val[j] = val;
                    diag_idx.push(row_idx.len() as I - 1);
                    has_q_diag = true;
                }
            }
            // Add the identity part for dx if unavailable
            if !has_q_diag {
                row_idx.push(j);
                values.push(1.0);
                diag_val[j] = 0.0;
                diag_idx.push(row_idx.len() as I - 1);
            }

            for idx in a_col_ptrs[j]..a_col_ptrs[j + 1] {
                let i = a_row_idx[idx];
                let val = a_values[idx];
                row_idx.push(i);
                values.push(val);
            }
        }

        let a_csr = qp.A.to_row_major().unwrap();
        let a_row_ptr = a_csr.symbolic().row_ptr();
        let a_col_idx = a_csr.symbolic().col_idx();
        let a_values = a_csr.val();

        // Create [ A^T ]
        for j in 0..n_con {
            col_ptrs.push(row_idx.len() as I);
            for idx in a_row_ptr[j]..a_row_ptr[j + 1] {
                let i = a_col_idx[idx];
                let val = a_values[idx];
                row_idx.push(i);
                values.push(val);
            }
        }

        col_ptrs.push(row_idx.len() as I);

        self.mat = unsafe {
            let sym = SymbolicSparseColMat::new_unchecked(
                n_var + n_con,
                n_var + n_con,
                col_ptrs,
                None,
                row_idx,
            );
            SparseColMat::<I, E>::new(sym, values)
        };

        // Initialize the linear solver with the KKT matrix
        self.solver.analyze(self.mat.as_ref()).unwrap();
        self.diag_idx = diag_idx;
        self.diag_val = diag_val;
    }

    fn solve(&mut self, view: &View<P, Workspace>) -> Delta {
        let View {
            program: qp,
            state,
            work,
        } = view;

        // Construct the augmented RHS
        let (n, m) = (state.variables().x().nrows(), state.variables().y().nrows());
        let SolverState {
            vars, residuals: _, ..
        } = state;
        let Workspace { rhs, .. } = work;

        let z_l_xl_inv = zip!(&vars.z_l(), &vars.x(), &qp.l)
            .map(|unzip!(z_l_i, x_i, l_i)| *z_l_i / (*x_i - *l_i));
        let z_u_xu_inv = zip!(&vars.z_u(), &vars.x(), &qp.u)
            .map(|unzip!(z_u_i, x_i, u_i)| *z_u_i / (*x_i - *u_i));

        // Update diagonal entries of the KKT matrix based on the current state
        let sys_diag = zip!(&self.diag_val, &z_l_xl_inv, &z_u_xu_inv).map(
            |unzip!(diag_val_i, z_l_xl_inv_i, z_u_xu_inv_i)| {
                *diag_val_i + *z_l_xl_inv_i + *z_u_xu_inv_i
            },
        );
        self.diag_idx
            .iter()
            .zip(sys_diag.iter())
            .for_each(|(idx, val)| {
                self.mat.val_mut()[*idx as usize] = *val;
            });

        let mut rhs_aug = Col::zeros(n + m);

        let (mut rhs_dual, mut rhs_primal) = rhs_aug.split_at_row_mut(n);

        let dz_l_rhs = zip!(&rhs.dz_l(), &vars.x(), &qp.l)
            .map(|unzip!(z_l_i, x_i, l_i)| *z_l_i / (*x_i - *l_i));
        let dz_u_rhs = zip!(&rhs.dz_u(), &vars.x(), &qp.u)
            .map(|unzip!(z_u_i, x_i, u_i)| *z_u_i / (*x_i - *u_i));

        rhs_dual.copy_from(rhs.dx().as_ref() + dz_l_rhs.as_ref() + dz_u_rhs.as_ref());
        rhs_primal.copy_from(rhs.dy().as_ref());

        // Solve the KKT system using the linear solver
        let solution = self.solver.solve(rhs_aug.as_mat()).unwrap();
        let solution = solution.col(0);

        // Apply the solution to the workspace
        let mut delta = state.delta.clone();
        let (delta_x, delta_y) = solution.split_at_row(n);
        delta.dx_mut().copy_from(delta_x);
        delta.dy_mut().copy_from(delta_y);
        delta.dz_l_mut().copy_from(
            zip!(&delta_x, &z_l_xl_inv, &dz_l_rhs)
                .map(|unzip!(delta_x_i, z_l_xl_inv_i, dz_l_rhs_i)| {
                    *dz_l_rhs_i - *delta_x_i * *z_l_xl_inv_i
                })
                .as_ref(),
        );
        delta.dz_u_mut().copy_from(
            zip!(&delta_x, &z_u_xu_inv, &dz_u_rhs)
                .map(|unzip!(delta_x_i, z_u_xu_inv_i, dz_u_rhs_i)| {
                    *dz_u_rhs_i - *delta_x_i * *z_u_xu_inv_i
                })
                .as_ref(),
        );
        delta
    }

    fn factorize_sys(&mut self, view: &View<P, Workspace>) {
        let View {
            program: qp,
            state,
            work: _,
        } = view;

        // Update matrix
        let vars = state.variables();
        let sys_diag = zip!(
            &self.diag_val,
            &vars.x(),
            &qp.l,
            &qp.u,
            &vars.z_l(),
            &vars.z_u()
        )
        .map(|unzip!(diag_val_i, x_i, l_i, u_i, z_l_i, z_u_i)| {
            *diag_val_i + *z_l_i / (*x_i - *l_i) + *z_u_i / (*x_i - *u_i)
        });

        self.diag_idx
            .iter()
            .zip(sys_diag.iter())
            .for_each(|(idx, val)| {
                self.mat.val_mut()[*idx as usize] = *val;
            });

        // Factorize the matrix
        self.solver.factorize(self.mat.as_ref()).unwrap();
    }
}
