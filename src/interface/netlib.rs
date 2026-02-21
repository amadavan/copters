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
use sif_rs::SIF;

use crate::{E, I, lp::LinearProgram};

pub trait TryFromSIF {
    fn try_into_linear_program(self) -> Result<LinearProgram, Problem>;
}

impl TryFromSIF for SIF {
    fn try_into_linear_program(self) -> Result<LinearProgram, Problem> {        
        // Map variable and constraint names to their respective internal indices
        // Use BTreeSet/BTreeMap for deterministic ordering of indices
        let map_var_idx: std::collections::BTreeMap<_, _> = self.get_cols().into_iter().map(|(var_name, _)| var_name.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().enumerate().map(|(i, var_name)| (var_name, i)).collect();
        let map_con_idx: std::collections::BTreeMap<_, _> = self.get_rows().into_iter().filter(|(_, rhs_type)| rhs_type != &&sif_rs::types::RowType::N).map(|(con_name, _)| con_name.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().enumerate().map(|(i, con_name)| (con_name, i)).collect();

        let (n_var, n_con) = (map_var_idx.len(), map_con_idx.len());

        // Get number of slack variables
        let n_slack = self.get_rows().iter().filter(|(_, rhs_type)| **rhs_type == sif_rs::types::RowType::L || **rhs_type == sif_rs::types::RowType::G).count();

        // Construct the objective function
        let mut c = Col::zeros(n_var + n_slack);
        self.get_entries().iter().filter(|((con, _var), _)| 
                // Filter out non-objective function coefficients
                self.get_rows().get(con) == Some(&&sif_rs::types::RowType::N))
            .for_each(|((_con, var), &val)| {
                let j = map_var_idx[var];
                c[j] = E::from(val);
            });

        // Construct the right-hand side vector
        let b = self.get_rhs().into_iter().filter(|(con, _val)| self.get_rows().get(*con) != Some(&&sif_rs::types::RowType::N)).map(|(con, val)| {
            let i = map_con_idx[con];
            (i, val)
        }).fold(Col::zeros(n_con), |mut b, (i, val)| {
            b[i] = E::from(*val);
            b
        });

        let a_triplets = self.get_entries().iter().filter(|((con, _var), val)| {
                // Filter out zero coefficients and objective function coefficients
                if **val == 0. {
                    return false;
                }

                if self.get_rows().get(con) == Some(&&sif_rs::types::RowType::N) {
                    return false;
                }

                true
            }).map(|(i, &val)| {
                let (i, j) = (map_con_idx[&i.0], map_var_idx[&i.1]);
                Triplet::new(I::from(i), I::from(j), E::from(val))
            }).collect::<Vec<_>>();

        // Construct bounds
        let mut l = Col::<E>::zeros(n_var + n_slack);
        let mut u = E::INFINITY * Col::<E>::ones(n_var + n_slack);
        self.get_bounds().into_iter().for_each(|(var_name, (bound_type, val))| {
                let j = map_var_idx[var_name];

                match bound_type {
                    sif_rs::types::BoundType::Lo => {
                        l[j] = E::from(*val);
                    }
                    sif_rs::types::BoundType::Up => {
                        u[j] = E::from(*val);
                    }
                    sif_rs::types::BoundType::Fr => {
                        l[j] = -E::INFINITY;
                        u[j] = E::INFINITY;
                    }
                    sif_rs::types::BoundType::Mi => {
                        l[j] = -E::INFINITY;
                        u[j] = E::from(0.);
                    }
                    sif_rs::types::BoundType::Pl => {
                        l[j] = E::from(0.);
                        u[j] = E::INFINITY;
                    }
                    sif_rs::types::BoundType::Fx => {
                        // TODO: cannot currently handle fixed variables properly because we need to ensure the initial iterate is strictly feasible. For now, we just add a small tolerance around the fixed value.
                        l[j] = E::from(*val - 0.01);
                        u[j] = E::from(*val + 0.01);
                    }
                    // sif_rs::types::BoundType::Bv => {
                    //     l[j] = E::from(0.);
                    //     u[j] = E::from(1.);
                    // }
                    // sif_rs::types::BoundType::Li => {
                    //     l[j] = E::from(*val);
                    //     u[j] = E::INFINITY;
                    // }
                    // sif_rs::types::BoundType::Ui => {
                    //     l[j] = -E::INFINITY;
                    //     u[j] = E::from(*val);
                    // }
                    // sif_rs::types::BoundType::Sc => {
                    //     // Special case for semi-continuous variables: either 0 or above a threshold
                    //     l[j] = E::from(*val);
                    //     u[j] = E::INFINITY;
                    // }
                    _ => panic!("Unsupported bound type: {:?}", bound_type),
                }
            });

        // Add slack variable coefficients to the constraint matrix
        let slack_triplets = map_con_idx.iter()
            .map(| (con_name, &i)| (self.get_rows()[con_name], i))
            .filter(|(con_type, _)| *con_type == sif_rs::types::RowType::L || *con_type == sif_rs::types::RowType::G)
            .enumerate()
            .map(|(i, (con_type, j))| {
                match con_type {
                    sif_rs::types::RowType::L => Triplet::new(I::from(j), I::from(n_var + i), E::from(1.)),
                    sif_rs::types::RowType::G => Triplet::new(I::from(j), I::from(n_var + i), E::from(-1.)),
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
    use super::*;

    use faer::Col;
    use rstest::rstest;
    use rstest_reuse::{apply, template};

    use crate::{
        E, SolverHooks, SolverOptions, SolverState, callback::{ConvergenceOutput}, lp::{LinearProgram, LPSolverType}, terminators::{ConvergenceTerminator, Terminator}
    };

    #[template]
    #[rstest]
    pub fn netlib_cases(
        #[values(
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
            LPSolverType::MpcSimplicialCholesky,
            LPSolverType::MpcSupernodalCholesky,
            LPSolverType::MpcSimplicialLu
        )]
        solver_type: LPSolverType,) {}

    #[apply(netlib_cases)]
    // #[apply(solver_types)]
    fn test_netlib_case(case_name: &str, solver_type: LPSolverType) {
        let lp = loaders::sif::netlib::get_case(case_name).unwrap().try_into_linear_program().unwrap();

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

        let mut properties = SolverHooks {
            callback: Box::new(ConvergenceOutput::new()),
            terminator: Box::new(ConvergenceTerminator::new(&options)),
        };

        let mut solver = LinearProgram::solver_builder(&lp)
            .with_solver(solver_type)
            .build()
            .unwrap();
        let status = solver.solve(&mut state, &mut properties);

        assert_eq!(status.unwrap(), crate::Status::Optimal);
    }

    #[test]
    pub fn test_lp_from_sif_afiro() {
        let lp = loaders::sif::netlib::get_case("afiro").unwrap().try_into_linear_program().unwrap();

        assert_eq!(lp.get_n_vars(), 32 + 19); // 32 original variables + 19 slack variables for inequalities
        assert_eq!(lp.get_n_cons(), 27);
    }
}
