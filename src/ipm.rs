use faer::{ColMut, ColRef};

use crate::{E, state, state::Residuals};

pub(crate) const DEFAULT_MAX_ITERATIONS: usize = 1000;

#[derive(Debug, Clone)]
pub struct Workspace {
    mu: E,
    sigma: E,
    tau: E,
    rhs: RHS,
}

impl state::Workspace for Workspace {}

impl Workspace {
    pub fn mu(&self) -> E {
        self.mu
    }

    pub fn set_mu(&mut self, value: E) {
        self.mu = value;
    }

    pub fn sigma(&self) -> E {
        self.sigma
    }

    pub fn set_sigma(&mut self, value: E) {
        self.sigma = value;
    }

    pub fn tau(&self) -> E {
        self.tau
    }

    pub fn set_tau(&mut self, value: E) {
        self.tau = value;
    }

    pub fn rhs(&self) -> &RHS {
        &self.rhs
    }

    pub fn rhs_mut(&mut self) -> &mut RHS {
        &mut self.rhs
    }
}

#[derive(Debug, Clone)]
pub struct RHS {
    data: Vec<E>,
    n: usize,
    m: usize,
}

impl RHS {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            data: vec![E::from(0.); (n + m + n + n) as usize],
            n,
            m,
        }
    }

    pub fn dual(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[0..self.m])
    }

    pub fn dual_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[0..self.m])
    }

    pub fn primal(&self) -> ColRef<'_, E> {
        ColRef::from_slice(&self.data[self.m..(self.n + self.m)])
    }

    pub fn primal_mut(&mut self) -> ColMut<'_, E> {
        ColMut::from_slice_mut(&mut self.data[self.m..(self.n + self.m)])
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

    pub fn copy_from_residual(&mut self, residual: &Residuals) {
        self.data.copy_from_slice(residual.get_raw());
    }
}
