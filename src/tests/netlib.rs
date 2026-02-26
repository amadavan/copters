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

use faer::Col;
use rstest::rstest;
use rstest_reuse::{apply, template};

use crate::{
    E, SolverHooks, SolverOptions, SolverState,
    callback::ConvergenceOutput,
    interface::sif::TryFromSIF,
    lp::{LPSolverType, LinearProgram},
    qp::{QPSolverType, QuadraticProgram},
    terminators::{ConvergenceTerminator, Terminator},
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
) {
}

#[apply(netlib_cases)]
fn lp(
    case_name: &str,
    #[values(
        LPSolverType::MpcSimplicialCholesky,
        LPSolverType::MpcSupernodalCholesky,
        LPSolverType::MpcSimplicialLu
    )]
    solver_type: LPSolverType,
) {
    let lp =
        LinearProgram::try_from_sif(&loaders::sif::netlib::get_case(case_name).unwrap()).unwrap();

    let mut state = SolverState::new(
        Col::ones(lp.get_n_vars()),
        Col::ones(lp.get_n_cons()),
        Col::ones(lp.get_n_vars()),
        -Col::<E>::ones(lp.get_n_vars()),
    );

    // Ensure that x is strictly between bounds for the initial iterate
    for (j, (l, u)) in lp
        .get_lower_bounds()
        .iter()
        .zip(lp.get_upper_bounds().iter())
        .enumerate()
    {
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

#[apply(netlib_cases)]
fn qp(
    case_name: &str,
    #[values(
        QPSolverType::MpcSimplicialCholesky,
        QPSolverType::MpcSupernodalCholesky,
        // QPSolverType::MpcSimplicialLu
    )]
    solver_type: QPSolverType,
) {
    let qp = QuadraticProgram::try_from_sif(&loaders::sif::netlib::get_case(case_name).unwrap())
        .unwrap();

    let mut state = SolverState::new(
        Col::ones(qp.get_n_vars()),
        Col::ones(qp.get_n_cons()),
        Col::ones(qp.get_n_vars()),
        -Col::<E>::ones(qp.get_n_vars()),
    );

    // Ensure that x is strictly between bounds for the initial iterate
    for (j, (l, u)) in qp
        .get_lower_bounds()
        .iter()
        .zip(qp.get_upper_bounds().iter())
        .enumerate()
    {
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

    let mut solver = QuadraticProgram::solver_builder(&qp)
        .with_solver(solver_type)
        .build()
        .unwrap();
    let status = solver.solve(&mut state, &mut properties);

    assert_eq!(status.unwrap(), crate::Status::Optimal);
}
