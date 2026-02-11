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
        lp::mpc::line_search::LPLineSearch,
    >
);
solver_builder!(
    MPCSupernodal,
    MehrotraPredictorCorrector<
        'static,
        SupernodalSparseCholesky,
        lp::mpc::augmented_system::StandardSystem<'static, SupernodalSparseCholesky>,
        lp::mpc::mu_update::AdaptiveMuUpdate<'static>,
        lp::mpc::line_search::LPLineSearch,
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
            let _lp = loaders::netlib::NetlibLoader::get_lp($name)?;
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
            let _lp = loaders::netlib::NetlibLoader::get_lp(stringify!($case))?;
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
    israel,
    scagr7,
    ship08s,
    vtp_base,
    bnl1,
    pilot,
    standgub,
    scsd1,
    sc205,
    adlittle,
    ship04s,
    scfxm2,
    agg,
    agg3,
    scorpion,
    shell,
    greenbeb,
    fit2d,
    bandm,
    share2b,
    sc105,
    nesm,
    boeing2,
    sc50b,
    scfxm3,
    stair,
    stocfor1,
    maros,
    bore3d,
    scsd8,
    stocfor2,
    fv47_25 = "25fv47",
    sctap1,
    ship12l,
    beaconfd,
    modszk1,
    cycle,
    ship12s,
    forplan,
    kb2,
    recipe,
    fit1d,
    e226,
    etamacro,
    perold,
    fffff800,
    sierra,
    maros_r7,
    tuff,
    pilotnov,
    dfl001,
    pilot87,
    pilot_we,
    capri,
    pilot4,
    wood1p,
    woodw,
    ship04l,
    grow15,
    degen3,
    fit1p,
    standata,
    greenbea,
    czprob,
    scfxm1,
    sc50a,
    agg2,
    standmps,
    share1b,
    afiro,
    seba,
    degen2,
    scagr25,
    scrs8,
    ganges,
    brandy,
    scsd6,
    boeing1,
    grow7,
    bnl2,
    sctap3,
    pilot_ja,
    blend,
    sctap2,
    d6cube,
    grow22,
    gfrd_pnc,
    ship08l,
    d2q06c,
    lotfi,
    finnis
);
