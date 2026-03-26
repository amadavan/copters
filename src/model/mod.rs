use std::{
    collections::HashMap,
    hash::Hash,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

use crate::{E, I, nlp::NonlinearProgram, state::Status};

pub mod expr;

pub type LinExpr = expr::LinExpr;
pub type QuadExpr = expr::QuadExpr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjSense {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstrType {
    Equality,
    LessThan,
    GreaterThan,
    Range,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId(I);

#[derive(Debug, Clone)]
pub struct Var {
    name: String,
    pub lb: E,
    pub ub: E,
    idx: VarId,
}

impl Var {
    pub fn name(&self) -> &str {
        &self.name
    }

    fn set_lb(&mut self, lb: E) {
        self.lb = lb;
    }

    fn set_ub(&mut self, ub: E) {
        self.ub = ub;
    }

    fn idx(&self) -> VarId {
        self.idx
    }
}

impl Hash for Var {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Var {}

#[derive(Debug, Clone)]
struct VarPool {
    vars: Vec<Var>,
}

impl VarPool {
    fn new() -> Self {
        Self { vars: Vec::new() }
    }

    fn add_var(&mut self, lb: E, ub: E) -> VarId {
        let var = Var {
            name: format!("var{}", self.vars.len()),
            lb,
            ub,
            idx: VarId(self.vars.len()),
        };
        self.vars.push(var);
        VarId(self.vars.len() - 1)
    }

    fn get_var(&self, idx: VarId) -> Option<&Var> {
        self.vars.get(idx.0 as usize)
    }

    fn get_var_mut(&mut self, idx: VarId) -> Option<&mut Var> {
        self.vars.get_mut(idx.0 as usize)
    }

    fn get_var_by_name(&self, name: &str) -> Option<&Var> {
        self.vars.iter().find(|var| var.name() == name)
    }
}

impl Mul<E> for VarId {
    type Output = LinExpr;

    fn mul(self, rhs: E) -> Self::Output {
        LinExpr::new(vec![(self, rhs)], E::from(0.))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstrId(I);

#[derive(Debug, Clone)]
pub struct LinConstr {
    name: String,
    pub expr: LinExpr,
    pub constr_type: ConstrType,
    pub lb: E,
    pub ub: E,
    idx: ConstrId,
}

impl LinConstr {
    pub fn expr(&self) -> &LinExpr {
        &self.expr
    }

    pub fn expr_mut(&mut self) -> &mut LinExpr {
        &mut self.expr
    }

    pub fn constr_type(&self) -> ConstrType {
        self.constr_type
    }

    pub fn set_constr_type(&mut self, constr_type: ConstrType) {
        self.constr_type = constr_type;
    }

    pub fn lb(&self) -> E {
        self.lb
    }

    pub fn set_lb(&mut self, lb: E) {
        self.lb = lb;
    }

    pub fn ub(&self) -> E {
        self.ub
    }

    pub fn set_ub(&mut self, ub: E) {
        self.ub = ub;
    }

    fn idx(&self) -> ConstrId {
        self.idx
    }
}

struct LinConstrPool {
    constrs: Vec<LinConstr>,
}

impl LinConstrPool {
    fn new() -> Self {
        Self {
            constrs: Vec::new(),
        }
    }

    fn add_constr(&mut self, expr: LinExpr, constr_type: ConstrType, lb: E, ub: E) -> ConstrId {
        let constr = LinConstr {
            name: format!("constr{}", self.constrs.len()),
            expr,
            constr_type,
            lb,
            ub,
            idx: ConstrId(self.constrs.len() as I),
        };
        self.constrs.push(constr);
        ConstrId(self.constrs.len() as I - 1)
    }

    fn get_constr(&self, idx: ConstrId) -> Option<&LinConstr> {
        self.constrs.get(idx.0 as usize)
    }

    fn get_constr_mut(&mut self, idx: ConstrId) -> Option<&mut LinConstr> {
        self.constrs.get_mut(idx.0 as usize)
    }

    fn get_constr_by_name(&self, name: &str) -> Option<&LinConstr> {
        self.constrs.iter().find(|constr| constr.name == name)
    }
}

pub struct LinearModel {
    var_pool: VarPool,
    constr_pool: LinConstrPool,

    vars: Vec<VarId>,
    sense: ObjSense,
    lin_obj: LinExpr,
    constrs: Vec<ConstrId>,
}

impl LinearModel {
    pub fn new() -> Self {
        Self {
            var_pool: VarPool::new(),
            constr_pool: LinConstrPool::new(),

            vars: Vec::new(),
            sense: ObjSense::Minimize,
            lin_obj: LinExpr::default(),
            constrs: Vec::new(),
        }
    }

    pub fn vars(&self) -> Vec<&Var> {
        self.vars
            .iter()
            .filter_map(|&var_id| self.var_pool.get_var(var_id))
            .collect()
    }

    pub fn add_var(&mut self, lb: E, ub: E) -> VarId {
        let var_id = self.var_pool.add_var(lb, ub);
        self.vars.push(var_id);
        var_id
    }

    pub fn get_var(&self, idx: VarId) -> Option<&Var> {
        self.var_pool.get_var(idx)
    }

    pub fn get_var_mut(&mut self, idx: VarId) -> Option<&mut Var> {
        self.var_pool.get_var_mut(idx)
    }

    pub fn get_var_by_name(&self, name: &str) -> Option<&Var> {
        self.var_pool.get_var_by_name(name)
    }

    pub fn remove_var(&mut self, var: VarId) {
        if let Some(pos) = self.vars.iter().position(|&var_id| var_id == var) {
            let _ = self.vars.remove(pos);

            self.constrs.iter().for_each(|&constr_id| {
                if let Some(constr) = self.constr_pool.get_constr_mut(constr_id) {
                    constr
                        .expr_mut()
                        .coeffs_mut()
                        .retain(|&(v_id, _)| v_id != var);
                }
            });

            self.lin_obj.coeffs_mut().retain(|&(v_id, _)| v_id != var);
        }
    }

    pub fn constrs(&self) -> Vec<&LinConstr> {
        self.constrs
            .iter()
            .filter_map(|&constr_id| self.constr_pool.get_constr(constr_id))
            .collect()
    }

    pub fn add_constr(&mut self, expr: LinExpr, constr_type: ConstrType, lb: E, ub: E) -> ConstrId {
        let constr_id = self.constr_pool.add_constr(expr, constr_type, lb, ub);
        self.constrs.push(constr_id);
        constr_id
    }

    pub fn get_constr(&self, idx: ConstrId) -> Option<&LinConstr> {
        self.constr_pool.get_constr(idx)
    }

    pub fn get_constr_mut(&mut self, idx: ConstrId) -> Option<&mut LinConstr> {
        self.constr_pool.get_constr_mut(idx)
    }

    pub fn remove_constr(&mut self, constr: ConstrId) {
        if let Some(pos) = self
            .constrs
            .iter()
            .position(|&constr_id| constr_id == constr)
        {
            self.constrs.remove(pos);
        }
    }

    pub fn get_constr_by_name(&self, name: &str) -> Option<&LinConstr> {
        self.constr_pool.get_constr_by_name(name)
    }

    pub fn objective(&self) -> &LinExpr {
        &self.lin_obj
    }

    pub fn objective_mut(&mut self) -> &mut LinExpr {
        &mut self.lin_obj
    }

    pub fn set_objective(&mut self, obj: LinExpr) {
        self.lin_obj = obj;
    }

    pub fn get_sense(&self) -> ObjSense {
        self.sense
    }

    pub fn set_sense(&mut self, sense: ObjSense) {
        self.sense = sense;
    }
}

pub type NonlinearModel = NonlinearProgram;

pub struct Solution {
    status: Status,
    objective: E,
    values: Vec<E>,
    duals: Vec<E>,
    rcs: Vec<E>,
}

impl Solution {
    pub fn status(&self) -> Status {
        self.status
    }

    pub fn objective(&self) -> E {
        self.objective
    }

    pub fn values(&self) -> &Vec<E> {
        &self.values
    }

    pub fn get_value(&self, var: &Var) -> Option<E> {
        self.values.get(var.idx.0 as usize).cloned()
    }

    pub fn duals(&self) -> &Vec<E> {
        &self.duals
    }

    pub fn get_dual(&self, constr: &LinConstr) -> Option<E> {
        self.duals.get(constr.name.parse::<usize>().ok()?).cloned()
    }

    pub fn reduced_costs(&self) -> &Vec<E> {
        &self.rcs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lin_constr() {
        let var1 = Var {
            name: "x".to_string(),
            lb: 0.,
            ub: 10.,
            idx: VarId(0),
        };
        let constr = LinConstr {
            name: "constr0".to_string(),
            expr: LinExpr::from(var1.idx) * 1.,
            constr_type: ConstrType::LessThan,
            lb: E::from(0.),
            ub: E::from(5.),
            idx: ConstrId(0),
        };
        assert_eq!(constr.expr().coeffs(), &vec![(VarId(0), 1.)]);
        assert_eq!(constr.expr().constant(), 0.);
        assert_eq!(constr.constr_type(), ConstrType::LessThan);
        assert_eq!(constr.lb(), 0.);
        assert_eq!(constr.ub(), 5.);
    }

    #[test]
    fn test_linear_model() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(0., 10.);
        let var2 = model.add_var(-5., 5.);
        model.set_objective(LinExpr::from(var1) + LinExpr::from(var2));
        let mut constr = model.add_constr(
            LinExpr::from(var1) - LinExpr::from(var2),
            ConstrType::GreaterThan,
            0.,
            E::from(10.),
        );
        assert_eq!(model.vars().len(), 2);
        assert_eq!(model.constrs().len(), 1);
        assert_eq!(
            {
                let mut coeffs = model.objective().coeffs().clone();
                coeffs.sort_by_key(|(var_id, _)| var_id.0);
                coeffs
            },
            vec![(VarId(0), 1.), (VarId(1), 1.)]
        );
    }
}
