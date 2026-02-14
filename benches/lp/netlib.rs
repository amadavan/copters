use copters::linalg::cholesky::SimplicialSparseCholesky;
use copters::linalg::cholesky::SupernodalSparseCholesky;
use copters::lp;
use copters::lp::{LinearProgramSolver, mpc::MehrotraPredictorCorrector};

trait SolverBuilder {
    type Solver: LinearProgramSolver<'static>;
}

macro_rules! solver_builder {
    ($name:ident, $solver_type:ty) => {
        struct $name;

        impl SolverBuilder for $name {
            type Solver = $solver_type;
        }
    };
}

// Add any additional solvers to benchmark here
solver_builder!(
    MPCSimplicial,
    MehrotraPredictorCorrector<
        'static,
        SimplicialSparseCholesky,
        lp::mpc::augmented_system::StandardSystem<'static, SimplicialSparseCholesky>,
        lp::mpc::mu_update::AdaptiveMuUpdate<'static>,
    >
);
solver_builder!(
    MPCSupernodal,
    MehrotraPredictorCorrector<
        'static,
        SupernodalSparseCholesky,
        lp::mpc::augmented_system::StandardSystem<'static, SupernodalSparseCholesky>,
        lp::mpc::mu_update::AdaptiveMuUpdate<'static>,
    >
);

macro_rules! netlib_benches {
    (@bench $case:ident = $name:literal) => {
        #[divan::bench(
            types = [
                MPCSimplicial,
                MPCSupernodal,
            ]
        )]
        fn $case<S: SolverBuilder>() -> Result<(), String> {
            let _lp = loaders::netlib::get_case($name)?;
            Ok(())
        }
    };
    (@bench $case:ident) => {
        #[divan::bench(
            types = [
                MPCSimplicial,
                MPCSupernodal,
            ]
        )]
        fn $case<S: SolverBuilder>() -> Result<(), String> {
            let _lp = loaders::netlib::get_case(stringify!($case))?;
            Ok(())
        }
    };
    ($($case:ident $(= $name:literal)?),* $(,)?) => {
        $(
            netlib_benches!(@bench $case $(= $name)?);
        )*
    };
}

// Add netlib cases to consider
netlib_benches!(
    adlittle,
    afiro,
    agg,
    agg2,
    agg3,
    bandm,
    beaconfd,
    blend,
    bnl1,
    bnl2,
    boeing1,
    boeing2,
    bore3d,
    brandy,
    capri,
    cycle,
    czprob,
    d2q06c,
    d6cube,
    degen2,
    degen3,
    dfl001,
    e226,
    etamacro,
    fffff800,
    finnis,
    fit1d,
    fit1p,
    fit2d,
    forplan,
    fv47_25 = "25fv47",
    ganges,
    gfrd_pnc,
    greenbea,
    greenbeb,
    grow15,
    grow22,
    grow7,
    israel,
    kb2,
    lotfi,
    maros_r7,
    maros,
    modszk1,
    nesm,
    perold,
    pilot_ja,
    pilot_we,
    pilot,
    pilot4,
    pilot87,
    pilotnov,
    recipe,
    sc105,
    sc205,
    sc50a,
    sc50b,
    scagr25,
    scagr7,
    scfxm1,
    scfxm2,
    scfxm3,
    scorpion,
    scrs8,
    scsd1,
    scsd6,
    scsd8,
    sctap1,
    sctap2,
    sctap3,
    seba,
    share1b,
    share2b,
    shell,
    ship04l,
    ship04s,
    ship08l,
    ship08s,
    ship12l,
    ship12s,
    sierra,
    stair,
    standata,
    standgub,
    standmps,
    stocfor1,
    stocfor2,
    tuff,
    vtp_base,
    wood1p,
    woodw,
);
