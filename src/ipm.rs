use faer::Col;

use crate::{E, SolverState};

#[allow(unused)]
#[derive(Debug, Clone, PartialEq)]
pub struct RHS {
    r_d: Col<E>,
    r_c: Col<E>,
    r_l: Col<E>,
    r_u: Col<E>,
}

impl From<&SolverState> for RHS {
    fn from(value: &SolverState) -> Self {
        Self {
            r_d: value.dual_feasibility.clone(),
            r_c: value.primal_feasibility.clone(),
            r_l: value.cs_lower.clone(),
            r_u: value.cs_upper.clone(),
        }
    }
}

// impl From<&mut SolverState> for RHS {
//     fn from(value: &mut SolverState) -> Self {
//         Self {
//             r_d: value.dual_feasibility.clone(),
//             r_c: value.primal_feasibility.clone(),
//             r_l: value.cs_lower.clone(),
//             r_u: value.cs_upper.clone(),
//         }
//     }
// }

impl RHS {
    pub fn r_d(&self) -> &Col<E> {
        &self.r_d
    }
    pub fn r_d_mut(&mut self) -> &mut Col<E> {
        &mut self.r_d
    }
    pub fn set_r_d(&mut self, value: Col<E>) {
        self.r_d = value;
    }

    pub fn r_c(&self) -> &Col<E> {
        &self.r_c
    }
    pub fn r_c_mut(&mut self) -> &mut Col<E> {
        &mut self.r_c
    }
    pub fn set_r_c(&mut self, value: Col<E>) {
        self.r_c = value;
    }

    pub fn r_l(&self) -> &Col<E> {
        &self.r_l
    }
    pub fn r_l_mut(&mut self) -> &mut Col<E> {
        &mut self.r_l
    }
    pub fn set_r_l(&mut self, value: Col<E>) {
        self.r_l = value;
    }

    pub fn r_u(&self) -> &Col<E> {
        &self.r_u
    }
    pub fn r_u_mut(&mut self) -> &mut Col<E> {
        &mut self.r_u
    }
    pub fn set_r_u(&mut self, value: Col<E>) {
        self.r_u = value;
    }
}
