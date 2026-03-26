#![allow(unused)]
use faer::{Col, ColMut, ColRef};

use crate::{E, OptimizationProgram, linalg};

/// Marker trait for algorithm-specific workspace data held in a [`View`].
pub trait Workspace: Clone {
    fn new<'a, P: OptimizationProgram>(program: &'a P, state: &'a mut SolverState) -> Self;
}

/// Bundles a mutable reference to the solver state with algorithm-specific workspace data.
#[derive(Debug)]
pub struct View<'a, P: OptimizationProgram, W: Workspace> {
    program: &'a P,
    state: &'a mut SolverState,
    work: W,
}

impl<'a, P: OptimizationProgram, W: Workspace> View<'a, P, W> {
    pub fn new(program: &'a P, state: &'a mut SolverState) -> Self {
        let work = W::new(program, state);
        Self {
            program,
            state,
            work,
        }
    }

    pub fn get_program(&self) -> &P {
        &self.program
    }

    pub fn state(&self) -> &SolverState {
        self.state
    }

    pub fn state_mut(&mut self) -> &mut SolverState {
        &mut self.state
    }

    pub fn variables(&self) -> &Variables {
        self.state.variables()
    }

    pub fn residuals(&self) -> &Residuals {
        self.state.residuals()
    }

    pub fn work(&self) -> &W {
        &self.work
    }

    pub fn work_mut(&mut self) -> &mut W {
        &mut self.work
    }
}

/// Status codes for optimization solvers.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Status {
    #[default]
    /// The solver is still running.
    InProgress,
    /// An optimal solution was found.
    Optimal,
    /// The problem is infeasible.
    Infeasible,
    /// The problem is unbounded.
    Unbounded,
    /// The status is unknown or not determined.
    Unknown,
    /// The solver stopped due to a time limit.
    TimeLimit,
    /// The solver stopped due to an iteration limit.
    IterationLimit,
    /// The solver was interrupted (e.g., by user or signal).
    Interrupted,
}

/// Current iterate: primal/dual variables and KKT residuals.
#[derive(Debug, Clone)]
pub struct SolverState {
    vars: Variables,
    residuals: Residuals,
    delta: Delta,

    nit: usize,
    status: Status,
    alpha_primal: E,
    alpha_dual: E,
}

impl SolverState {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            vars: Variables::new(n, m),
            residuals: Residuals::new(n, m),
            delta: Delta::new(n, m),
            nit: 0,
            status: Status::InProgress,
            alpha_primal: E::from(1.),
            alpha_dual: E::from(1.),
        }
    }

    pub fn variables(&self) -> &Variables {
        &self.vars
    }

    pub fn variables_mut(&mut self) -> &mut Variables {
        &mut self.vars
    }

    pub fn residuals(&self) -> &Residuals {
        &self.residuals
    }

    pub fn residuals_mut(&mut self) -> &mut Residuals {
        &mut self.residuals
    }

    pub fn delta(&self) -> &Delta {
        &self.delta
    }

    pub fn delta_mut(&mut self) -> &mut Delta {
        &mut self.delta
    }

    pub fn nit(&self) -> usize {
        self.nit
    }

    pub fn inc_nit(&mut self) {
        self.nit += 1;
    }

    pub fn set_nit(&mut self, nit: usize) {
        self.nit = nit;
    }

    pub fn set_status(&mut self, status: Status) {
        self.status = status;
    }

    pub fn status(&self) -> Status {
        self.status
    }

    pub fn alpha_primal(&self) -> E {
        self.alpha_primal
    }

    pub fn alpha_dual(&self) -> E {
        self.alpha_dual
    }

    pub fn alpha(&self) -> (E, E) {
        (self.alpha_primal, self.alpha_dual)
    }

    pub fn set_alpha(&mut self, alpha_primal: E, alpha_dual: E) {
        self.alpha_primal = alpha_primal;
        self.alpha_dual = alpha_dual;
    }
}

/// Primal and dual variables stored contiguously as `[x | y | z_l | z_u]`.
///
/// - `x`: primal (n)
/// - `y`: equality multipliers (m)
/// - `z_l`, `z_u`: bound multipliers (n each)
#[derive(Debug, Clone)]
pub struct Variables {
    data: Vec<E>,
    n: usize,
    m: usize,
}

impl Variables {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            data: vec![E::from(0.); (n + m + n + n) as usize],
            n,
            m,
        }
    }

    pub fn get_n(&self) -> usize {
        self.n
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn x(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[0..self.n])
    }

    pub fn x_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[0..self.n])
    }

    pub fn y(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[self.n..(self.n + self.m)])
    }

    pub fn y_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[self.n..(self.n + self.m)])
    }

    pub fn z_l(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn z_l_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn z_u(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m + self.n)..])
    }

    pub fn z_u_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[(self.n + self.m + self.n)..])
    }

    /// Reduced costs: `z_l - z_u`.
    pub fn rc(&self) -> Col<E> {
        &self.z_l() - &self.z_u()
    }

    /// Applies a Newton step: `vars += alpha * delta`.
    pub fn update(&mut self, alpha_primal: E, alpha_dual: E, delta: &Delta) {
        linalg::vector_ops::axpy(alpha_primal, delta.dx(), self.x_mut());
        linalg::vector_ops::axpy(alpha_dual, delta.dy(), self.y_mut());
        linalg::vector_ops::axpy(alpha_dual, delta.dz_l(), self.z_l_mut());
        linalg::vector_ops::axpy(alpha_dual, delta.dz_u(), self.z_u_mut());
    }

    pub fn get_raw(&self) -> &Vec<E> {
        &self.data
    }
}

/// KKT residuals stored contiguously as `[dual | primal | slack_l | slack_u]`.
#[derive(Debug, Clone)]
pub struct Residuals {
    data: Vec<E>,
    n: usize,
    m: usize,
}

impl Residuals {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            data: vec![E::from(0.); (n + m + n + n) as usize],
            n,
            m,
        }
    }

    pub fn get_n(&self) -> usize {
        self.n
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn dual(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[0..self.n])
    }

    pub fn dual_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[0..self.n])
    }

    pub fn primal(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[self.n..(self.n + self.m)])
    }

    pub fn primal_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[self.n..(self.n + self.m)])
    }

    pub fn slack_l(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn slack_l_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn slack_u(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m + self.n)..])
    }

    pub fn slack_u_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[(self.n + self.m + self.n)..])
    }

    pub fn get_raw(&self) -> &Vec<E> {
        &self.data
    }
}

/// Newton step directions `[dx | dy | dz_l | dz_u]`, matching the layout of `Variables`.
#[derive(Debug, Clone)]
pub struct Delta {
    data: Vec<E>,
    n: usize,
    m: usize,
}

impl Delta {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            data: vec![E::from(0.); (n + m + n + n) as usize],
            n,
            m,
        }
    }

    pub fn get_n(&self) -> usize {
        self.n
    }

    pub fn get_m(&self) -> usize {
        self.m
    }

    pub fn dx(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[0..self.n])
    }

    pub fn dx_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[0..self.n])
    }

    pub fn dy(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[self.n..(self.n + self.m)])
    }

    pub fn dy_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[self.n..(self.n + self.m)])
    }

    pub fn dz_l(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn dz_l_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn dz_u(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m + self.n)..])
    }

    pub fn dz_u_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[(self.n + self.m + self.n)..])
    }

    pub fn get_raw(&self) -> &Vec<E> {
        &self.data
    }
}
