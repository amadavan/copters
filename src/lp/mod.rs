use faer::{
    Col,
    sparse::{SparseColMat, Triplet},
};

use crate::{E, I, IterativeSolver, SolverOptions};

pub mod mpc;

/// A linear program in standard form:
///
/// ```text
/// min  c^T x
/// s.t. A x = b
///      l <= x <= u
/// ```
#[allow(non_snake_case)]
pub struct LinearProgram {
    /// Objective function coefficients.
    c: Col<E>,
    /// Constraint matrix (sparse, column-major).
    A: SparseColMat<I, E>,
    /// Right-hand side of the equality constraints.
    b: Col<E>,
    /// Lower bounds on the variables.
    l: Col<E>,
    /// Upper bounds on the variables.
    u: Col<E>,
}

#[allow(non_snake_case)]
impl LinearProgram {
    /// Creates a new linear program from the objective, constraints, and bounds.
    pub fn new(c: Col<E>, A: SparseColMat<I, E>, b: Col<E>, l: Col<E>, u: Col<E>) -> Self {
        Self { c, A, b, l, u }
    }

    /// Returns the number of variables (columns of `A`).
    pub fn get_n_vars(&self) -> usize {
        self.c.nrows()
    }

    /// Returns the number of constraints (rows of `A`).
    pub fn get_n_cons(&self) -> usize {
        self.b.nrows()
    }

    /// Returns `(n_vars, n_cons)`.
    pub fn get_dims(&self) -> (usize, usize) {
        (self.get_n_vars(), self.get_n_cons())
    }
}

/// Converts an MPS model into a [`LinearProgram`].
///
/// Inequality constraints (`<=` / `>=`) are converted to equalities by
/// introducing non-negative slack variables. The objective row (`Nr` type)
/// is separated from the constraint rows.
impl From<mps::model::Model<f32>> for LinearProgram {
    fn from(model: mps::model::Model<f32>) -> Self {
        // Get the bound type
        let rhs_type = {
            model
                .row_types
                .0
                .iter()
                .collect::<std::collections::HashMap<_, _>>()
        };
        
        // Map variable and constraint names to their respective internal indices
        let map_var_idx: std::collections::HashMap<_, _> = model
            .values.0.iter()
            .map(|((_, var), _)| var.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(i, var_name)| (var_name, i))
            .collect();
        let map_con_idx: std::collections::HashMap<_, _> = model
            .values.0.iter()
            .map(|((con, _), _)| con.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .filter(|con_name| rhs_type.get(con_name) != Some(&&mps::types::RowType::Nr)) // Filter out objective function row
            .enumerate()
            .map(|(i, con_name)| (con_name, i))
            .collect();

        let (n_var, n_con) = (map_var_idx.len(), map_con_idx.len());

        // Construct the objective function
        let mut c = Col::zeros(n_var);
        model
            .values
            .0
            .iter()
            .filter(|((con, var), _)| 
                // Filter out non-objective function coefficients
                rhs_type.get(con) == Some(&&mps::types::RowType::Nr))
            .for_each(|((con, var), &val)| {
                let j = map_var_idx[var];
                c[j] = E::from(val);
            });

        // Construct the right-hand side vector
        let mut b = Col::zeros(n_con);
        model
            .values
            .0
            .iter()
            .filter(|((con, _var), _)| {
                // Filter out objective function coefficients
                rhs_type.get(con) != Some(&&mps::types::RowType::Nr)
            })
            .for_each(|((con, _var), &val)| {
                let i = map_con_idx[con];
                b[i] = E::from(val);
            });

        // Construct the constraint matrix in triplet form
        let a_triplets = model
            .values
            .0
            .iter()
            .filter(|((con, _var), val)| {
                // Filter out zero coefficients and objective function coefficients
                if **val == 0. {
                    return false;
                }

                if rhs_type.get(con) == Some(&&mps::types::RowType::Nr) {
                    return false;
                }

                true
            })
            .map(|(i, &val)| {
                let (i, j) = (map_con_idx[&i.0], map_var_idx[&i.1]);
                Triplet::new(I::from(i), I::from(j), E::from(val))
            })
            .collect::<Vec<_>>();

        // Construct bounds
        let mut l = -E::INFINITY * Col::<E>::ones(n_var);
        let mut u = E::INFINITY * Col::<E>::ones(n_var);
        model.bounds.0.iter().for_each(|(var, bound)| {
            // We don't care about variable bound names
            bound.iter().for_each(|((var_name, bound_type), val)| {
                let j = map_var_idx[var_name];

                match bound_type {
                    mps::types::BoundType::Lo => {
                        l[j] = E::from(val.unwrap());
                    }
                    mps::types::BoundType::Up => {
                        u[j] = E::from(val.unwrap());
                    }
                    mps::types::BoundType::Fr => {
                        l[j] = -E::INFINITY;
                        u[j] = E::INFINITY;
                    }
                    mps::types::BoundType::Mi => {
                        l[j] = -E::INFINITY;
                        u[j] = E::from(val.unwrap());
                    }
                    mps::types::BoundType::Pl => {
                        l[j] = E::from(val.unwrap());
                        u[j] = E::INFINITY;
                    }

                    _ => panic!("Unsupported bound type: {:?}", bound_type),
                }
            });
        });
        
        // Add slack variables for inequalities
        let n_slack = model
            .row_types
            .0
            .iter()
            .filter(|(_, row_type)| **row_type == mps::types::RowType::Leq || **row_type == mps::types::RowType::Geq)
            .count();

        // Extend c, l, u for slack variables
        c.resize_with(n_var + n_slack, |_i| E::from(0.));
        l.resize_with(n_var + n_slack, |_i| E::from(0.));
        u.resize_with(n_var + n_slack, |_i| E::from(0.));

        // Add slack variable coefficients to the constraint matrix
        let slack_triplets = map_con_idx.iter()
            .map(| (con_name, &i)| (rhs_type[con_name], i))
            .filter(|(con_type, _)| **con_type == mps::types::RowType::Leq || **con_type == mps::types::RowType::Geq)
            .enumerate()
            .map(|(i, (con_type, j))| {
                match con_type {
                    mps::types::RowType::Leq => Triplet::new(I::from(j), I::from(n_var + i), E::from(1.)),
                    mps::types::RowType::Geq => Triplet::new(I::from(j), I::from(n_var + i), E::from(-1.)),
                    _ => unreachable!(),
                }
            });

        let a_triplets = a_triplets.into_iter().chain(slack_triplets).collect::<Vec<_>>();

        let A = SparseColMat::try_new_from_triplets(n_con, n_var + n_slack, a_triplets.as_slice()).unwrap();

        Self { c, A, b, l, u }
    }
}

/// Trait for solvers that operate on a [`LinearProgram`].
pub trait LinearProgramSolver<'a>: IterativeSolver {
    /// Creates a new solver instance for the given linear program and options.
    fn new(lp: &'a LinearProgram, options: &SolverOptions) -> Self;
}

#[cfg(test)]
mod test {
    use std::sync::OnceLock;

    use faer::{
        Col,
        sparse::{SparseColMat, Triplet},
    };
    use macros::matrix_parameterized_test;

    use crate::{
        E, I, Properties, SolverOptions, SolverState,
        callback::{Callback, ConvergenceOutput},
        linalg::cholesky::SimplicialSparseCholesky,
        lp::{LinearProgram, LinearProgramSolver, mpc},
        terminators::{ConvergenceTerminator, Terminator},
    };

    fn build_simple_lp() -> &'static LinearProgram {
        static LP: OnceLock<LinearProgram> = OnceLock::new();
        LP.get_or_init(|| {
            let a_triplets: [Triplet<I, I, E>; 9] = [
                Triplet::new(0, 0, -1.),
                Triplet::new(1, 0, 1.),
                Triplet::new(2, 0, -1.),
                Triplet::new(0, 1, -1.),
                Triplet::new(1, 1, -2.),
                Triplet::new(2, 1, 1.),
                Triplet::new(2, 2, 1.),
                Triplet::new(0, 3, 1.),
                Triplet::new(1, 4, 1.),
            ];
            let a = SparseColMat::try_new_from_triplets(3, 5, a_triplets.as_slice()).unwrap();

            LinearProgram::new(
                Col::from_fn(5, |i| [2., 1., 0., 0., 0.][i]),
                a,
                Col::from_fn(3, |i| [-2., 4., 1.][i]),
                Col::from_fn(5, |i| [-E::INFINITY, 0., 0., 0., 0.][i]),
                Col::from_fn(5, |i| {
                    [
                        E::INFINITY,
                        E::INFINITY,
                        E::INFINITY,
                        E::INFINITY,
                        E::INFINITY,
                    ][i]
                }),
            )
        })
    }

    type MPCSimplicial<'a> = mpc::MehrotraPredictorCorrector<
        'a,
        SimplicialSparseCholesky,
        mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
        mpc::mu_update::AdaptiveMuUpdate<'a>,
        mpc::line_search::LPLineSearch<'a>,
    >;
    type MPCSupernodal<'a> = mpc::MehrotraPredictorCorrector<
        'a,
        SimplicialSparseCholesky,
        mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
        mpc::mu_update::AdaptiveMuUpdate<'a>,
        mpc::line_search::LPLineSearch<'a>,
    >;

    #[matrix_parameterized_test(
        types = (MPCSimplicial, MPCSupernodal),
        named_args = [("Simple LP", build_simple_lp()),]
    )]
    fn test_lp<'a, T: LinearProgramSolver<'a>>(lp: &'a LinearProgram) {
        let mut state = SolverState::new(
            Col::ones(lp.c.nrows()),
            Col::ones(lp.b.nrows()),
            Col::ones(lp.c.nrows()),
            -Col::<E>::ones(lp.c.nrows()),
        );

        let mut options = SolverOptions::new();

        let mut properties = Properties {
            callback: Box::new(ConvergenceOutput::new(&options)),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut s = T::new(lp, &options);
        let status = s.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }

    #[test]
    pub fn test_lp_from_mps_afiro() {
        let model = loaders::netlib::get_lp("afiro").unwrap();
        let lp = LinearProgram::from(model);

        assert_eq!(lp.get_n_vars(), 32 + 19); // 32 original variables + 19 slack variables for inequalities
        assert_eq!(lp.get_n_cons(), 27);
    }
}
