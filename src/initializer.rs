use faer::{unzip, zip};

use crate::{E, OptimizationProgram, SolverOptions, qp::QuadraticProgram, state::SolverState};

pub trait Initializer {
    type Program: OptimizationProgram;

    fn initialize(
        &mut self,
        program: &Self::Program,
        state: &mut SolverState,
        _options: &SolverOptions,
    ) {
        // Default implementation does nothing, but can be overridden by specific solvers
    }
}

pub struct PrimalDualFeasible<P: OptimizationProgram> {}

impl Initializer for PrimalDualFeasible<QuadraticProgram> {
    type Program = QuadraticProgram;

    fn initialize(
        &mut self,
        program: &QuadraticProgram,
        state: &mut SolverState,
        _options: &SolverOptions,
    ) {
        // Initialize primal variables to the midpoint of the bounds
        zip!(state.variables_mut().x_mut(), &program.l, &program.u).for_each(
            |unzip!(x, lb, ub)| {
                if lb.is_finite() && ub.is_finite() {
                    *x = (lb + ub) / E::from(2.);
                } else if lb.is_finite() {
                    *x = lb + E::from(1.);
                } else if ub.is_finite() {
                    *x = ub - E::from(1.);
                } else {
                    *x = E::from(0.);
                }
            },
        );

        // Initialize dual variables to 1
        state.variables_mut().y_mut().fill(0.);
        state.variables_mut().z_l_mut().fill(1.);
        state.variables_mut().z_u_mut().fill(-1.);
    }
}

pub struct Mehrotra {}
