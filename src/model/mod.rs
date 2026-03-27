use std::{collections::HashMap, hash::Hash, ops::Mul};

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
pub struct Var<'a> {
    pub name: &'a str,
    pub lb: &'a E,
    pub ub: &'a E,
    idx: VarId,
}

impl Var<'_> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn lb(&self) -> E {
        *self.lb
    }

    pub fn ub(&self) -> E {
        *self.ub
    }

    #[allow(unused)]
    fn idx(&self) -> VarId {
        self.idx
    }
}

pub struct VarMut<'a> {
    name: &'a str,
    pub lb: &'a mut E,
    pub ub: &'a mut E,
    idx: VarId,
}

impl VarMut<'_> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn lb(&mut self) -> &mut E {
        self.lb
    }

    pub fn ub(&mut self) -> &mut E {
        self.ub
    }

    #[allow(unused)]
    fn idx(&self) -> VarId {
        self.idx
    }
}

#[derive(Debug, Clone)]
struct VarArena {
    lbs: Vec<E>,
    ubs: Vec<E>,
    name_lookup: HashMap<String, VarId>,
}

impl VarArena {
    fn new() -> Self {
        Self {
            lbs: Vec::new(),
            ubs: Vec::new(),
            name_lookup: HashMap::new(),
        }
    }

    fn add_var(&mut self, name: String, lb: E, ub: E) -> VarId {
        let idx = self.lbs.len();
        self.lbs.push(lb);
        self.ubs.push(ub);
        self.name_lookup.insert(name, VarId(idx as I));
        VarId(idx as I)
    }

    fn get_var<'a>(&'a self, idx: VarId) -> Option<Var<'a>> {
        if idx.0 < self.lbs.len() as I {
            Some(Var {
                name: self
                    .name_lookup
                    .iter()
                    .find(|(_, v)| **v == idx)
                    .map(|(n, _)| n.as_str())
                    .unwrap_or(""),
                lb: &self.lbs[idx.0 as usize],
                ub: &self.ubs[idx.0 as usize],
                idx,
            })
        } else {
            None
        }
    }

    fn get_var_mut<'a>(&'a mut self, idx: VarId) -> Option<VarMut<'a>> {
        if idx.0 < self.lbs.len() as I {
            let name = self
                .name_lookup
                .iter()
                .find(|(_, v)| **v == idx)
                .map(|(n, _)| n.as_str());
            if let Some(name) = name {
                Some(VarMut {
                    name,
                    lb: &mut self.lbs[idx.0 as usize],
                    ub: &mut self.ubs[idx.0 as usize],
                    idx,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_var_by_name<'a>(&'a self, name: &str) -> Option<Var<'a>> {
        self.name_lookup
            .get(name)
            .and_then(|&idx| self.get_var(idx))
    }

    fn with_var_mut<F, R>(&mut self, var_id: VarId, f: F) -> Option<R>
    where
        F: FnOnce(&mut VarMut) -> R,
    {
        self.get_var_mut(var_id).as_mut().map(|var_mut| f(var_mut))
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
pub struct LinConstr<'a> {
    name: &'a String,
    pub expr: &'a LinExpr,
    pub constr_type: &'a ConstrType,
    pub lb: &'a E,
    pub ub: &'a E,
    idx: ConstrId,
}

impl LinConstr<'_> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn expr(&self) -> &LinExpr {
        self.expr
    }

    pub fn constr_type(&self) -> ConstrType {
        *self.constr_type
    }

    pub fn lb(&self) -> E {
        *self.lb
    }

    pub fn ub(&self) -> E {
        *self.ub
    }

    #[allow(unused)]
    fn idx(&self) -> ConstrId {
        self.idx
    }
}

pub struct LinConstrMut<'a> {
    name: &'a String,
    pub expr: &'a mut LinExpr,
    pub constr_type: &'a mut ConstrType,
    pub lb: &'a mut E,
    pub ub: &'a mut E,
    idx: ConstrId,
}

impl LinConstrMut<'_> {
    pub fn name(&self) -> &str {
        self.name
    }

    pub fn expr(&self) -> &LinExpr {
        self.expr
    }

    pub fn expr_mut(&mut self) -> &mut LinExpr {
        self.expr
    }

    pub fn constr_type(&self) -> ConstrType {
        *self.constr_type
    }

    pub fn set_constr_type(&mut self, constr_type: ConstrType) {
        *self.constr_type = constr_type;
    }

    pub fn lb(&self) -> E {
        *self.lb
    }

    pub fn set_lb(&mut self, lb: E) {
        *self.lb = lb;
    }

    pub fn ub(&self) -> E {
        *self.ub
    }

    pub fn set_ub(&mut self, ub: E) {
        *self.ub = ub;
    }

    #[allow(unused)]
    fn idx(&self) -> ConstrId {
        self.idx
    }
}

struct LinConstrArena {
    exprs: Vec<LinExpr>,
    constr_types: Vec<ConstrType>,
    lbs: Vec<E>,
    ubs: Vec<E>,
    name_lookup: HashMap<String, ConstrId>,
}

impl LinConstrArena {
    fn new() -> Self {
        Self {
            exprs: Vec::new(),
            constr_types: Vec::new(),
            lbs: Vec::new(),
            ubs: Vec::new(),
            name_lookup: HashMap::new(),
        }
    }

    fn add_constr(
        &mut self,
        name: String,
        expr: LinExpr,
        constr_type: ConstrType,
        lb: E,
        ub: E,
    ) -> ConstrId {
        let idx = self.exprs.len();
        self.name_lookup.insert(name.clone(), ConstrId(idx as I));
        self.exprs.push(expr);
        self.constr_types.push(constr_type);
        self.lbs.push(lb);
        self.ubs.push(ub);
        ConstrId(idx as I)
    }

    fn get_constr<'a>(&'a self, idx: ConstrId) -> Option<LinConstr<'a>> {
        if idx.0 < self.exprs.len() as I {
            let name = self
                .name_lookup
                .iter()
                .find(|(_, v)| **v == idx)
                .map(|(n, _)| n);
            if let Some(name) = name {
                Some(LinConstr {
                    name: name,
                    expr: &self.exprs[idx.0 as usize],
                    constr_type: &self.constr_types[idx.0 as usize],
                    lb: &self.lbs[idx.0 as usize],
                    ub: &self.ubs[idx.0 as usize],
                    idx,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_constr_mut<'a>(&'a mut self, idx: ConstrId) -> Option<LinConstrMut<'a>> {
        if idx.0 < self.exprs.len() as I {
            let name = self
                .name_lookup
                .iter()
                .find(|(_, v)| **v == idx)
                .map(|(n, _)| n);
            if let Some(name) = name {
                Some(LinConstrMut {
                    name: name,
                    expr: &mut self.exprs[idx.0 as usize],
                    constr_type: &mut self.constr_types[idx.0 as usize],
                    lb: &mut self.lbs[idx.0 as usize],
                    ub: &mut self.ubs[idx.0 as usize],
                    idx,
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    fn get_constr_by_name<'a>(&'a self, name: &str) -> Option<LinConstr<'a>> {
        self.name_lookup
            .get(name)
            .and_then(|&idx| self.get_constr(idx))
    }

    fn with_constr_mut<F, R>(&mut self, constr_id: ConstrId, f: F) -> Option<R>
    where
        F: FnOnce(&mut LinConstrMut) -> R,
    {
        self.get_constr_mut(constr_id)
            .as_mut()
            .map(|constr_mut| f(constr_mut))
    }
}

pub struct LinearModel {
    var_arena: VarArena,
    constr_arena: LinConstrArena,

    pub vars: Vec<VarId>,
    pub sense: ObjSense,
    pub lin_obj: LinExpr,
    pub constrs: Vec<ConstrId>,
}

impl LinearModel {
    pub fn new() -> Self {
        Self {
            var_arena: VarArena::new(),
            constr_arena: LinConstrArena::new(),

            vars: Vec::new(),
            sense: ObjSense::Minimize,
            lin_obj: LinExpr::default(),
            constrs: Vec::new(),
        }
    }

    pub fn vars(&self) -> Vec<VarId> {
        self.vars.clone()
    }

    pub fn add_var(&mut self, name: Option<String>, lb: E, ub: E) -> VarId {
        let var_id = self.var_arena.add_var(
            name.unwrap_or_else(|| format!("var{}", self.vars.len())),
            lb,
            ub,
        );
        self.vars.push(var_id);
        var_id
    }

    pub fn get_var<'a>(&'a self, idx: VarId) -> Option<Var<'a>> {
        self.var_arena.get_var(idx)
    }

    pub fn get_var_mut<'a>(&'a mut self, idx: VarId) -> Option<VarMut<'a>> {
        self.var_arena.get_var_mut(idx)
    }

    pub fn get_var_by_name<'a>(&'a self, name: &str) -> Option<Var<'a>> {
        self.var_arena.get_var_by_name(name)
    }

    pub fn remove_var(&mut self, var: VarId) {
        if let Some(pos) = self.vars.iter().position(|&var_id| var_id == var) {
            let _ = self.vars.remove(pos);
        }
    }

    pub fn with_var_mut<F, R>(&mut self, var_id: VarId, f: F) -> Option<R>
    where
        F: FnOnce(&mut VarMut) -> R,
    {
        self.var_arena.with_var_mut(var_id, f)
    }

    pub fn constrs(&self) -> Vec<ConstrId> {
        self.constrs.clone()
    }

    pub fn add_constr(
        &mut self,
        name: Option<String>,
        expr: LinExpr,
        constr_type: ConstrType,
        lb: E,
        ub: E,
    ) -> ConstrId {
        let constr_id = self.constr_arena.add_constr(
            name.unwrap_or_else(|| format!("constr{}", self.constrs.len())),
            expr,
            constr_type,
            lb,
            ub,
        );
        self.constrs.push(constr_id);
        constr_id
    }

    pub fn get_constr<'a>(&'a self, idx: ConstrId) -> Option<LinConstr<'a>> {
        self.constr_arena.get_constr(idx)
    }

    pub fn get_constr_mut<'a>(&'a mut self, idx: ConstrId) -> Option<LinConstrMut<'a>> {
        self.constr_arena.get_constr_mut(idx)
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

    pub fn get_constr_by_name<'a>(&'a self, name: &str) -> Option<LinConstr<'a>> {
        self.constr_arena.get_constr_by_name(name)
    }

    pub fn with_constr_mut<F, R>(&mut self, constr_id: ConstrId, f: F) -> Option<R>
    where
        F: FnOnce(&mut LinConstrMut) -> R,
    {
        self.constr_arena.with_constr_mut(constr_id, f)
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
    pub status: Status,
    pub objective: E,
    pub values: Vec<E>,
    pub duals: Vec<E>,
    pub rcs: Vec<E>,
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
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);

        let constr = model.add_constr(None, LinExpr::from(var1) * 1., ConstrType::LessThan, 0., 5.);
        let constr = model.get_constr(constr).unwrap();

        assert_eq!(constr.expr().coeffs(), &vec![(VarId(0), 1.)]);
        assert_eq!(constr.expr().constant(), 0.);
        assert_eq!(constr.constr_type(), ConstrType::LessThan);
        assert_eq!(constr.lb(), 0.);
        assert_eq!(constr.ub(), 5.);
    }

    #[test]
    fn test_get_var() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(Some("x".to_string()), 0., 10.);
        let var2 = model.add_var(Some("y".to_string()), -5., 5.);

        let var_x = model.get_var(var1).unwrap();
        assert_eq!(var_x.name(), "x");
        assert_eq!(var_x.lb(), 0.);
        assert_eq!(var_x.ub(), 10.);
        assert_eq!(var_x.idx(), var1);

        let var_y = model.get_var(var2).unwrap();
        assert_eq!(var_y.name(), "y");
        assert_eq!(var_y.lb(), -5.);
        assert_eq!(var_y.ub(), 5.);
        assert_eq!(var_y.idx(), var2);
    }

    #[test]
    fn test_get_var_by_name() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(Some("x".to_string()), 0., 10.);
        let var2 = model.add_var(Some("y".to_string()), -5., 5.);

        let var_x = model.get_var_by_name("x").unwrap();
        assert_eq!(var_x.name(), "x");
        assert_eq!(var_x.lb(), 0.);
        assert_eq!(var_x.ub(), 10.);
        assert_eq!(var_x.idx(), var1);

        let var_y = model.get_var_by_name("y").unwrap();
        assert_eq!(var_y.name(), "y");
        assert_eq!(var_y.lb(), -5.);
        assert_eq!(var_y.ub(), 5.);
        assert_eq!(var_y.idx(), var2);
    }

    #[test]
    fn test_var_mut() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(Some("x".to_string()), 0., 10.);
        let mut var_mut = model.get_var_mut(var1).unwrap();

        assert_eq!(var_mut.name(), "x");
        assert_eq!(*var_mut.lb(), 0.);
        assert_eq!(*var_mut.ub(), 10.);
        assert_eq!(var_mut.idx(), var1);

        *var_mut.lb() = -5.;
        *var_mut.ub() = 15.;

        assert_eq!(*var_mut.lb(), -5.);
        assert_eq!(*var_mut.ub(), 15.);
    }

    #[test]
    fn test_with_var_mut() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(Some("x".to_string()), 0., 10.);

        model.with_var_mut(var1, |var_mut| {
            assert_eq!(var_mut.name(), "x");
            assert_eq!(*var_mut.lb(), 0.);
            assert_eq!(*var_mut.ub(), 10.);
            assert_eq!(var_mut.idx(), var1);

            *var_mut.lb() = -5.;
            *var_mut.ub() = 15.;

            assert_eq!(*var_mut.lb(), -5.);
            assert_eq!(*var_mut.ub(), 15.);
        });
    }

    #[test]
    fn test_get_constr() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let constr = model.add_constr(None, LinExpr::from(var1) * 1., ConstrType::LessThan, 0., 5.);
        let constr = model.get_constr(constr).unwrap();

        assert_eq!(constr.expr().coeffs(), &vec![(VarId(0), 1.)]);
        assert_eq!(constr.expr().constant(), 0.);
        assert_eq!(constr.constr_type(), ConstrType::LessThan);
        assert_eq!(constr.lb(), 0.);
        assert_eq!(constr.ub(), 5.);
    }

    #[test]
    fn test_get_constr_by_name() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let _constr = model.add_constr(
            Some("c1".to_string()),
            LinExpr::from(var1) * 1.,
            ConstrType::LessThan,
            0.,
            5.,
        );
        let constr = model.get_constr_by_name("c1").unwrap();

        assert_eq!(constr.expr().coeffs(), &vec![(VarId(0), 1.)]);
        assert_eq!(constr.expr().constant(), 0.);
        assert_eq!(constr.constr_type(), ConstrType::LessThan);
        assert_eq!(constr.lb(), 0.);
        assert_eq!(constr.ub(), 5.);
    }

    #[test]
    fn test_lin_constr_mut() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let constr = model.add_constr(None, LinExpr::from(var1) * 1., ConstrType::LessThan, 0., 5.);
        let mut constr_mut = model.get_constr_mut(constr).unwrap();

        assert_eq!(constr_mut.expr().coeffs(), &vec![(VarId(0), 1.)]);
        assert_eq!(constr_mut.expr().constant(), 0.);
        assert_eq!(constr_mut.constr_type(), ConstrType::LessThan);
        assert_eq!(constr_mut.lb(), 0.);
        assert_eq!(constr_mut.ub(), 5.);

        constr_mut.set_constr_type(ConstrType::GreaterThan);
        constr_mut.set_lb(-5.);
        constr_mut.set_ub(10.);

        assert_eq!(constr_mut.constr_type(), ConstrType::GreaterThan);
        assert_eq!(constr_mut.lb(), -5.);
        assert_eq!(constr_mut.ub(), 10.);
    }

    #[test]
    fn test_with_constr_mut() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let constr = model.add_constr(None, LinExpr::from(var1) * 1., ConstrType::LessThan, 0., 5.);
        model.with_constr_mut(constr, |constr_mut| {
            assert_eq!(constr_mut.expr().coeffs(), &vec![(VarId(0), 1.)]);
            assert_eq!(constr_mut.expr().constant(), 0.);
            assert_eq!(constr_mut.constr_type(), ConstrType::LessThan);
            assert_eq!(constr_mut.lb(), 0.);
            assert_eq!(constr_mut.ub(), 5.);

            constr_mut.set_constr_type(ConstrType::GreaterThan);
            constr_mut.set_lb(-5.);
            constr_mut.set_ub(10.);

            assert_eq!(constr_mut.constr_type(), ConstrType::GreaterThan);
            assert_eq!(constr_mut.lb(), -5.);
            assert_eq!(constr_mut.ub(), 10.);
        });
    }

    #[test]
    fn test_linear_model() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let var2 = model.add_var(None, -5., 5.);
        model.set_objective(LinExpr::from(var1) + LinExpr::from(var2));
        let _constr = model.add_constr(
            None,
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
