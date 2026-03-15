#![allow(unused)]
use faer::{Col, ColMut, ColRef};

use crate::{E, linalg};

/// Marker trait for algorithm-specific workspace data held in a [`View`].
pub trait Workspace {}

/// Bundles a mutable reference to the solver state with algorithm-specific workspace data.
#[derive(Debug, PartialEq)]
pub struct View<'a, W: Workspace> {
    state: &'a mut SolverState,
    work: W,
}

impl<'a, W: Workspace> View<'a, W> {
    pub fn new(state: &'a mut SolverState, work: W) -> Self {
        Self { state, work }
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

/// Current iterate: primal/dual variables and KKT residuals.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverState {
    vars: Variables,
    residuals: Residuals,
}

impl SolverState {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            vars: Variables::new(n, m),
            residuals: Residuals::new(n, m),
        }
    }

    pub fn variables(&self) -> &Variables {
        &self.vars
    }

    pub fn residuals(&self) -> &Residuals {
        &self.residuals
    }
}

/// Primal and dual variables stored contiguously as `[x | y | z_l | z_u]`.
///
/// - `x`: primal (n)
/// - `y`: equality multipliers (m)
/// - `z_l`, `z_u`: bound multipliers (n each)
#[derive(Debug, Clone, PartialEq)]
struct Variables {
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
#[derive(Debug, Clone, PartialEq)]
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

    pub fn primal(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[self.n..(self.n + self.m)])
    }

    pub fn slack_l(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn slack_u(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m + self.n)..])
    }

    pub fn get_raw(&self) -> &Vec<E> {
        &self.data
    }
}

/// Newton step directions `[dx | dy | dz_l | dz_u]`, matching the layout of `Variables`.
#[derive(Debug, Clone, PartialEq)]
struct Delta {
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

    pub fn dy(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[self.n..(self.n + self.m)])
    }

    pub fn dz_l(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m)..(self.n + self.m + self.n)])
    }

    pub fn dz_u(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[(self.n + self.m + self.n)..])
    }

    pub fn get_raw(&self) -> &Vec<E> {
        &self.data
    }
}
