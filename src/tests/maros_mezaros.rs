use faer::Col;
use rstest::rstest;
use rstest_reuse::{apply, template};

use crate::{
    E, SolverHooks, SolverOptions, SolverState,
    callback::ConvergenceOutput,
    interface::sif::TryFromSIF,
    qp::{QPSolverType, QuadraticProgram},
    terminators::{ConvergenceTerminator, Terminator},
};

#[template]
#[rstest]
pub fn maros_mezaros_cases(
    #[values(
        //
            // "AUG2D",
            "AUG2DC",
            // "AUG2DCQP", // Long
            // "AUG2DQP",  // Long
            // "AUG3D",
            "AUG3DC",
            "AUG3DCQP",
            "AUG3DQP",
            // "BOYD1",
            // "BOYD2",
            // "CONT-050",
            // "CONT-100",
            // "CONT-101",
            // "CONT-200",
            // "CONT-201",
            // "CONT-300",
            // "CVXQP1_L",
            // "CVXQP1_M",
            // "CVXQP1_S",
            // "CVXQP2_L",
            // "CVXQP2_M",
            // "CVXQP2_S",
            // "CVXQP3_L",
            // "CVXQP3_M",
            // "CVXQP3_S",
            // "DPKLO1",
            "DTOC3",
            // "DUAL1",
            // "DUAL2",
            // "DUAL3",
            // "DUAL4",
            // "DUALC1",
            // "DUALC2",
            // "DUALC5",
            // "DUALC8",
            // "EXDATA",
            // "GENHS28",
            // "GOULDQP2",
            // "GOULDQP3",
            // "HS118",
            // "HS21",
            // "HS268",
            // "HS35",
            // "HS35MOD",
            // "HS51",
            // "HS52",
            // "HS53",
            // "HS76",
            // "HUES-MOD",
            "HUESTIS",
            // "KSIP",
            // "LASER",
            // "LISWET1",
            // "LISWET10",
            // "LISWET11",
            // "LISWET12",
            // "LISWET2",
            // "LISWET3",
            // "LISWET4",
            // "LISWET5",
            // "LISWET6",
            // "LISWET7",
            // "LISWET8",
            // "LISWET9",
            // "LOTSCHD",
            // "MOSARQP1",
            // "MOSARQP2",
            // "POWELL20",
            // "PRIMAL1",
            // "PRIMAL2",
            // "PRIMAL3",
            // "PRIMAL4",
            // "PRIMALC1",
            // "PRIMALC2",
            // "PRIMALC5",
            // "PRIMALC8",
            // "Q25FV47",
            // "QADLITTL",
            // "QAFIRO",
            // "QBANDM",
            // "QBEACONF",
            // "QBORE3D",
            // "QBRANDY",
            // "QCAPRI",
            // "QE226",
            // "QETAMACR",
            // "QFFFFF80",
            // "QFORPLAN",
            // "QGFRDXPN",
            // "QGROW15",
            // "QGROW22",
            // "QGROW7",
            // "QISRAEL",
            // "QPCBLEND",
            // "QPCBOEI1",
            // "QPCBOEI2",
            // "QPCSTAIR",
            // "QPILOTNO",
            // "QPTEST",
            // "QRECIPE",
            // "QSC205",
            // "QSCAGR25",
            // "QSCAGR7",
            // "QSCFXM1",
            // "QSCFXM2",
            // "QSCFXM3",
            // "QSCORPIO",
            // "QSCRS8",
            // "QSCSD1",
            // "QSCSD6",
            // "QSCSD8",
            // "QSCTAP1",
            // "QSCTAP2",
            // "QSCTAP3",
            // "QSEBA",
            // "QSHARE1B",
            // "QSHARE2B",
            // "QSHELL",
            // "QSHIP04L",
            // "QSHIP04S",
            // "QSHIP08L",
            // "QSHIP08S",
            // "QSHIP12L",
            // "QSHIP12S",
            // "QSIERRA",
            // "QSTAIR",
            // "QSTANDAT",
            // "S268",
            // "STADAT1",
            // "STADAT2",
            // "STADAT3",
            // "STCQP1",
            // "STCQP2",
            "TAME",
            // "UBH1",
            // "VALUES",
            // "YAO",
            // "ZECEVIC2",
        )]
    case_name: &str,
    #[values(
        QPSolverType::MpcSimplicialCholesky,
        QPSolverType::MpcSupernodalCholesky,
        QPSolverType::MpcSimplicialLu
    )]
    solver_type: QPSolverType,
) {
}

#[apply(maros_mezaros_cases)]
fn test_case(case_name: &str, solver_type: QPSolverType) {
    let qp =
        QuadraticProgram::try_from_sif(&loaders::sif::maros_mezaros::get_case(case_name).unwrap())
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
