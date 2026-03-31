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

use faer::{unzip, zip};
use rstest::{fixture, rstest};
use rstest_reuse::{apply, template};

use crate::{
    E,
    SolverHooks,
    SolverOptions,
    SolverState,
    callback::ConvergenceOutput,
    data_loaders,
    interface::sif::TryFromSIF,
    // lp::{LPSolverType, LinearProgram},
    // nlp::{NLPSolverType, NonlinearProgram},
    qp::{self, QuadraticProgram},
    terminators::ConvergenceTerminator,
};

#[fixture]
fn download_cases() -> &'static () {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        data_loaders::sif::download_netlib_lp().unwrap();
    })
}

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

// #[apply(netlib_cases)]
// fn lp(
//     _download_cases: &(),
//     case_name: &str,
//     #[values(
//         LPSolverType::MpcSimplicialCholesky,
//         LPSolverType::MpcSupernodalCholesky,
//         LPSolverType::MpcSimplicialLu
//     )]
//     solver_type: LPSolverType,
// ) {
//     let lp = LinearProgram::try_from_sif(&data_loaders::sif::netlib::get_case(case_name).unwrap())
//         .unwrap();

//     let mut state = SolverState::new(lp.get_n_vars(), lp.get_n_cons());

//     // Ensure that x is strictly between bounds for the initial iterate
//     zip!(
//         state.variables_mut().x_mut(),
//         lp.get_lower_bounds(),
//         lp.get_upper_bounds()
//     )
//     .for_each(|unzip!(x_j, l, u)| {
//         if l.is_finite() && u.is_finite() {
//             *x_j = (l + u) / 2.;
//         } else if l.is_finite() && !u.is_finite() {
//             *x_j = l + 1.;
//         } else if !l.is_finite() && u.is_finite() {
//             *x_j = u - 1.;
//         } else {
//             *x_j = 0.;
//         }
//     });

//     let options = SolverOptions::new();

//     let mut properties = SolverHooks {
//         callback: Box::new(ConvergenceOutput::new(&options)),
//         terminator: Box::new(ConvergenceTerminator::new(&options)),
//     };

//     let mut solver = LinearProgram::solver_builder(&lp)
//         .with_solver(solver_type)
//         .build()
//         .unwrap();
//     let status = solver.solve(&mut state, &mut properties);

//     assert_eq!(status.unwrap(), crate::Status::Optimal);
// }

#[apply(netlib_cases)]
fn test_qp(
    case_name: &str,
    #[values(
        qp::SolverType::MpcSimplicialCholeskyDefault,
        qp::SolverType::MpcSupernodalCholeskyDefault,
        // qp::SolverType::PcSimplicialLu
    )]
    solver_type: qp::SolverType,
) {
    let qp =
        QuadraticProgram::try_from_sif(&data_loaders::sif::netlib::get_case(case_name).unwrap())
            .unwrap();

    let mut state = SolverState::new(qp.get_n_vars(), qp.get_n_cons());
    state.variables_mut().x_mut().fill(1.);
    state.variables_mut().y_mut().fill(1.);
    state.variables_mut().z_l_mut().fill(1.);
    state.variables_mut().z_u_mut().fill(-1.);

    // Ensure that x is strictly between bounds for the initial iterate
    zip!(state.variables_mut().x_mut(), &qp.l, &qp.u).for_each(|unzip!(x_j, l, u)| {
        if l.is_finite() && u.is_finite() {
            *x_j = (l + u) / 2.;
        } else if l.is_finite() && !u.is_finite() {
            *x_j = l + 1.;
        } else if !l.is_finite() && u.is_finite() {
            *x_j = u - 1.;
        } else {
            *x_j = 0.;
        }
    });

    let options = SolverOptions::new();

    let mut properties = SolverHooks {
        callback: Box::new(ConvergenceOutput::new(&options)),
        terminator: Box::new(ConvergenceTerminator::new(&options)),
    };

    let mut solver = qp::Builder::new()
        .with_qp(&qp)
        .with_solver(solver_type)
        .build()
        .unwrap();
    let status = solver.solve(&qp, &mut state, &mut properties);

    assert_eq!(status.unwrap(), crate::Status::Optimal);
}

// #[apply(netlib_cases)]
// fn nlp(case_name: &str, #[values(NLPSolverType::InteriorPointMethod)] solver_type: NLPSolverType) {
//     let qp =
//         QuadraticProgram::try_from_sif(&data_loaders::sif::netlib::get_case(case_name).unwrap())
//             .unwrap();
//     let nlp = (&qp).into();

//     let mut state = SolverState::new(nlp.get_n_vars(), nlp.get_n_cons());

//     zip!(
//         state.variables_mut().x_mut(),
//         nlp.get_lower_bounds(),
//         nlp.get_upper_bounds()
//     )
//     .for_each(|unzip!(x_j, l, u)| {
//         if l.is_finite() && u.is_finite() {
//             *x_j = (l + u) / 2.;
//         } else if l.is_finite() && !u.is_finite() {
//             *x_j = l + 1.;
//         } else if !l.is_finite() && u.is_finite() {
//             *x_j = u - 1.;
//         } else {
//             *x_j = 0.;
//         }
//     });

//     let options = SolverOptions::new();

//     let mut properties = SolverHooks {
//         callback: Box::new(ConvergenceOutput::new(&options)),
//         terminator: Box::new(ConvergenceTerminator::new(&options)),
//     };

//     let mut solver = NonlinearProgram::solver_builder(&nlp)
//         .with_solver(solver_type)
//         .build()
//         .unwrap();
//     let status = solver.solve(&mut state, &mut properties);

//     assert_eq!(status.unwrap(), crate::Status::Optimal);
// }
