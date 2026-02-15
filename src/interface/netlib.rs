//! Conversion from [MPS](https://en.wikipedia.org/wiki/MPS_(format)) models
//! (as parsed by the [`mps`] crate) into the internal [`LinearProgram`]
//! representation used by the solver.
//!
//! The conversion handles:
//! - Extracting the objective function from the `Nr`-typed row.
//! - Mapping named variables and constraints to contiguous indices
//!   (using [`BTreeSet`](std::collections::BTreeSet) for deterministic ordering).
//! - Introducing non-negative slack variables for inequality constraints
//!   (`<=` / `>=`).
//! - Translating MPS bound types (`Lo`, `Up`, `Fx`, `Fr`, `Mi`, `Pl`) into
//!   finite lower/upper bound vectors.

use faer::{Col, sparse::{SparseColMat, Triplet}};
use problemo::Problem;

use crate::{E, I, lp::LinearProgram};


/// Fallible conversion from an MPS model into a [`LinearProgram`].
///
/// This is intentionally a standalone trait rather than a [`TryFrom`] impl so
/// that it can live in this crate while both the MPS model type and
/// [`LinearProgram`] are defined elsewhere.
pub trait TryFromMpsModel {
    /// Consume the MPS model and produce the equivalent [`LinearProgram`] in
    /// standard form (`min c^T x  s.t.  Ax = b,  l <= x <= u`).
    fn try_into_linear_program(self) -> Result<LinearProgram, Problem>;
}

/// Converts an [`mps::model::Model<f32>`] into a [`LinearProgram`].
///
/// # Conversion steps
///
/// 1. **Index assignment** — Variable and constraint names are sorted
///    lexicographically ([`BTreeSet`](std::collections::BTreeSet)) and mapped to
///    contiguous `0..n` indices so the resulting matrices are deterministic
///    regardless of parse order.
/// 2. **Objective extraction** — Coefficients on the `Nr`-typed row become the
///    cost vector `c`; that row is excluded from the constraint matrix.
/// 3. **Constraint matrix & RHS** — Non-zero coefficients on non-`Nr` rows form
///    the sparse column-major matrix `A`, and the corresponding RHS values form
///    the vector `b`.
/// 4. **Bounds** — MPS bound records are translated into per-variable lower (`l`)
///    and upper (`u`) bound vectors. Default bounds are `0 <= x_j <= +inf`.
/// 5. **Slack variables** — Each `<=` constraint gets a `+1` slack column; each
///    `>=` constraint gets a `-1` slack column, converting inequalities to
///    equalities.
///
/// # Errors
///
/// Returns a [`Problem`] if the sparse matrix cannot be constructed from the
/// generated triplets (e.g. duplicate or out-of-bounds entries).
impl TryFromMpsModel for mps::model::Model<f32> {
    fn try_into_linear_program(self) -> Result<LinearProgram, Problem> {
        // Get the bound type
        let rhs_type = {
            self
                .row_types
                .0
                .iter()
                .collect::<std::collections::HashMap<_, _>>()
        };
        
        // Map variable and constraint names to their respective internal indices
        // Use BTreeSet/BTreeMap for deterministic ordering of indices
        let map_var_idx: std::collections::BTreeMap<_, _> = self
            .values.0.iter()
            .map(|((_, var), _)| var.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .enumerate()
            .map(|(i, var_name)| (var_name, i))
            .collect();
        let map_con_idx: std::collections::BTreeMap<_, _> = self
            .row_types.0.iter()
            .map(|(name, _)| name.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter(|con_name| rhs_type.get(con_name) != Some(&&mps::types::RowType::Nr)) // Filter out objective function row
            .enumerate()
            .map(|(i, con_name)| (con_name, i))
            .collect();

        let (n_var, n_con) = (map_var_idx.len(), map_con_idx.len());
               
        // Add slack variables for inequalities
        let n_slack = self
            .row_types
            .0
            .iter()
            .filter(|(_, row_type)| **row_type == mps::types::RowType::Leq || **row_type == mps::types::RowType::Geq)
            .count();

        // Construct the objective function
        let mut c = Col::zeros(n_var + n_slack);
        self
            .values
            .0
            .iter()
            .filter(|((con, _var), _)| 
                // Filter out non-objective function coefficients
                rhs_type.get(con) == Some(&&mps::types::RowType::Nr))
            .for_each(|((_con, var), &val)| {
                let j = map_var_idx[var];
                c[j] = E::from(val);
            });

        // Construct the right-hand side vector
        let b = {
            let mut b = Col::zeros(n_con);
            self
                .rhs
                .0
                .iter()
                .flat_map(|(_con, rhs)| {
                    // We don't care about constraint bound names
                    rhs.iter()
                })
                .filter(|(con, _val)| {
                    // Filter out objective function coefficients
                    rhs_type.get(con) != Some(&&mps::types::RowType::Nr)
                })
                .for_each(|(con, &val)| {
                    let i = map_con_idx[con];
                    b[i] = E::from(val);
                });
            b
        };

        // Construct the constraint matrix in triplet form
        let a_triplets = self
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
        let mut l = Col::<E>::zeros(n_var + n_slack);
        let mut u = E::INFINITY * Col::<E>::ones(n_var + n_slack);
        self.bounds.0.iter().flat_map(| (_name, map) | map.iter() ).for_each(|((var_name, bound_type), val)| {
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
                        u[j] = E::from(0.);
                    }
                    mps::types::BoundType::Pl => {
                        l[j] = E::from(0.);
                        u[j] = E::INFINITY;
                    }
                    mps::types::BoundType::Fx => {
                        // TODO: cannot currently handle fixed variables properly because we need to ensure the initial iterate is strictly feasible. For now, we just add a small tolerance around the fixed value.
                        l[j] = E::from(val.unwrap() - 0.01);
                        u[j] = E::from(val.unwrap() + 0.01);
                    }
                    // mps::types::BoundType::Bv => {
                    //     l[j] = E::from(0.);
                    //     u[j] = E::from(1.);
                    // }
                    // mps::types::BoundType::Li => {
                    //     l[j] = E::from(val.unwrap());
                    //     u[j] = E::INFINITY;
                    // }
                    // mps::types::BoundType::Ui => {
                    //     l[j] = -E::INFINITY;
                    //     u[j] = E::from(val.unwrap());
                    // }
                    // mps::types::BoundType::Sc => {
                    //     // Special case for semi-continuous variables: either 0 or above a threshold
                    //     l[j] = E::from(val.unwrap());
                    //     u[j] = E::INFINITY;
                    // }
                    _ => panic!("Unsupported bound type: {:?}", bound_type),
                }
            });

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

        #[allow(non_snake_case)]
        let A = SparseColMat::try_new_from_triplets(n_con, n_var + n_slack, a_triplets.as_slice()).unwrap();

        Ok(LinearProgram::new(c, A, b, l, u))
    }
}


#[cfg(test)]
mod test {
    use faer::Col;
    use rstest::rstest;
    use rstest_reuse::{apply, template};

    use loaders::netlib;

    use crate::{
        E, Properties, SolverOptions, SolverState, callback::{Callback, ConvergenceOutput}, interface::netlib::TryFromMpsModel, lp::{LinearProgramSolverBuilder, LinearProgramSolverType}, terminators::{ConvergenceTerminator, Terminator}
    };

    // fn get_netlib_case(name: &str) -> &'static LinearProgram {
    //     NETLIB_LPS.get(name).unwrap()
    // }
    
    // type MPCSimplicial<'a> = mpc::MehrotraPredictorCorrector<
    //     'a,
    //     SimplicialSparseCholesky,
    //     mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
    //     mpc::mu_update::AdaptiveMuUpdate<'a>,
    // >;
    // type MPCSupernodal<'a> = mpc::MehrotraPredictorCorrector<
    //     'a,
    //     SimplicialSparseCholesky,
    //     mpc::augmented_system::StandardSystem<'a, SimplicialSparseCholesky>,
    //     mpc::mu_update::AdaptiveMuUpdate<'a>,
    // >;
    
    #[template]
    #[rstest]
    pub fn solver_types(
        #[values(
            LinearProgramSolverType::SimplicialCholeskyMpc,
            LinearProgramSolverType::SupernodalCholeskyMpc,
            LinearProgramSolverType::SimplicialLuMpc
        )]
        solver_type: LinearProgramSolverType,
    ) {
    }

    #[template]
    #[rstest]
    pub fn netlib_cases(#[values(
            // "25fv47",
            // "adlittle",
            "afiro",
            // "agg",
            // "agg2",
            // "agg3",
            // "bandm",
            // "beaconfd",
            // "blend",
            // "bnl1",
            // "bnl2",
            // "boeing1",
            // "boeing2",
            // "bore3d",
            // "brandy",
            // "capri",
            // "cycle",
            // "czprob",
            // "d2q06c",
            // "d6cube",
            // "degen2",
            // "degen3",
            // "dfl001",
            // "e226",
            // "etamacro",
            // "fffff800",
            // "finnis",
            "fit1d",
            "fit1p",
            "fit2d",
            // "forplan",
            // "ganges",
            // "gfrd-pnc",
            // "greenbea",
            // "greenbeb",
            "grow15",
            "grow22",
            "grow7",
            // "israel",
            // "kb2",
            // "lotfi",
            // "maros-r7",
            // "maros",
            // "modszk1",
            // "nesm",
            // "perold",
            // "pilot.ja",
            // "pilot.we",
            // "pilot",
            // "pilot4",
            // "pilot87",
            // "pilotnov",
            // "recipe",
            // "sc105",
            // "sc205",
            // "sc50a",
            // "sc50b",
            // "scagr25",
            // "scagr7",
            // "scfxm1",
            // "scfxm2",
            // "scfxm3",
            // "scorpion",
            // "scrs8",
            "scsd1",
            "scsd6",
            "scsd8",
            "sctap1",
            "sctap2",
            "sctap3",
            // "seba",
            // "share1b",
            // "share2b",
            // "shell",
            // "ship04l",
            // "ship04s",
            // "ship08l",
            // "ship08s",
            // "ship12l",
            // "ship12s",
            // "sierra",
            // "stair",
            // "standata",
            // "standgub",
            // "standmps",
            // "stocfor1",
            // "stocfor2",
            // "tuff",
            // "vtp.base",
            "wood1p",
            // "woodw",
    )]
    case_name: &str, 
        #[values(
            LinearProgramSolverType::SimplicialCholeskyMpc,
            LinearProgramSolverType::SupernodalCholeskyMpc,
            LinearProgramSolverType::SimplicialLuMpc
        )]
        solver_type: LinearProgramSolverType,) {}

    #[apply(netlib_cases)]
    #[apply(solver_types)]
    fn test_netlib_case(case_name: &str, solver_type: LinearProgramSolverType) {
        let lp = netlib::get_case(case_name).unwrap().model().to_owned().try_into_linear_program().unwrap();

        let mut state = SolverState::new(
            Col::ones(lp.get_n_vars()),
            Col::ones(lp.get_n_cons()),
            Col::ones(lp.get_n_vars()),
            -Col::<E>::ones(lp.get_n_vars()),
        );

        // Ensure that x is strictly between bounds for the initial iterate
        for (j, (l, u)) in lp.get_lower_bounds().iter().zip(lp.get_upper_bounds().iter()).enumerate() {
            if l.is_finite() && u.is_finite() {
                state.x[j] = (l + u) / 2.;
            } else if l.is_finite() && !u.is_finite() {
                state.x[j] = l + 1.;
            } else if !l.is_finite() && u.is_finite() {
                state.x[j] = u - 1.;
            } else {
                state.x[j] = 0.;
            }
        }

        let options = SolverOptions::new();

        let mut properties = Properties {
            callback: Box::new(ConvergenceOutput::new(&options)),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut solver = LinearProgramSolverBuilder::new(&lp)
            .with_solver_type(solver_type)
            .build();
        let status = solver.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }

    #[test]
    pub fn test_lp_from_mps_afiro() {
        let model = loaders::netlib::get_case("afiro").unwrap().model().to_owned();
        let lp = model.try_into_linear_program().unwrap();

        assert_eq!(lp.get_n_vars(), 32 + 19); // 32 original variables + 19 slack variables for inequalities
        assert_eq!(lp.get_n_cons(), 27);
    }
}
