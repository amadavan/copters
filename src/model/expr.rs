use std::{
    collections::HashMap,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

use crate::{
    E,
    model::{Var, VarId},
};

#[derive(Debug, Clone)]
pub struct LinExpr {
    coeffs: Vec<(VarId, E)>,
    constant: E,
}

impl LinExpr {
    pub fn new(coeffs: Vec<(VarId, E)>, constant: E) -> Self {
        Self { coeffs, constant }
    }

    pub fn coeffs(&self) -> &Vec<(VarId, E)> {
        &self.coeffs
    }

    pub fn coeffs_mut(&mut self) -> &mut Vec<(VarId, E)> {
        &mut self.coeffs
    }

    pub fn constant(&self) -> E {
        self.constant
    }

    pub fn set_constant(&mut self, constant: E) {
        self.constant = constant;
    }

    pub fn sort(&mut self) {
        self.coeffs.sort_by_key(|(var_id, _)| var_id.0);
    }
}

impl Default for LinExpr {
    fn default() -> Self {
        Self {
            coeffs: Vec::new(),
            constant: E::from(0.),
        }
    }
}

impl From<E> for LinExpr {
    fn from(constant: E) -> Self {
        Self {
            coeffs: Vec::new(),
            constant,
        }
    }
}

impl From<VarId> for LinExpr {
    fn from(var: VarId) -> Self {
        LinExpr {
            coeffs: vec![(var, E::from(1.))],
            constant: E::from(0.),
        }
    }
}

impl From<&VarId> for LinExpr {
    fn from(var: &VarId) -> Self {
        LinExpr {
            coeffs: vec![(var.clone(), E::from(1.))],
            constant: E::from(0.),
        }
    }
}

impl Add for LinExpr {
    type Output = LinExpr;

    fn add(self, rhs: Self) -> Self::Output {
        let mut coeffs_map: HashMap<VarId, E> = HashMap::new();
        for (idx, coeff) in self.coeffs.into_iter().chain(rhs.coeffs.into_iter()) {
            *coeffs_map.entry(idx).or_insert(E::from(0.)) += coeff;
        }
        LinExpr {
            coeffs: coeffs_map.into_iter().collect(),
            constant: self.constant + rhs.constant,
        }
    }
}

impl AddAssign for LinExpr {
    fn add_assign(&mut self, rhs: Self) {
        let mut coeffs_map: HashMap<VarId, E> = self.coeffs.iter().cloned().collect();
        for (idx, coeff) in rhs.coeffs.into_iter() {
            *coeffs_map.entry(idx).or_insert(E::from(0.)) += coeff;
        }
        self.coeffs = coeffs_map.into_iter().collect();
        self.constant += rhs.constant;
    }
}

impl Sub for LinExpr {
    type Output = LinExpr;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (rhs * E::from(-1.))
    }
}

impl SubAssign for LinExpr {
    fn sub_assign(&mut self, rhs: Self) {
        *self += rhs * E::from(-1.);
    }
}

impl Mul<E> for LinExpr {
    type Output = LinExpr;

    fn mul(self, rhs: E) -> Self::Output {
        LinExpr {
            coeffs: self
                .coeffs
                .into_iter()
                .map(|(idx, coeff)| (idx, coeff * rhs))
                .collect(),
            constant: self.constant * rhs,
        }
    }
}

pub struct QuadExpr {
    lin_coeffs: Vec<(VarId, E)>,
    quad_coeffs: Vec<((VarId, VarId), E)>,
    constant: E,
}

impl QuadExpr {
    pub fn new(
        lin_coeffs: Vec<(VarId, E)>,
        quad_coeffs: Vec<((VarId, VarId), E)>,
        constant: E,
    ) -> Self {
        Self {
            lin_coeffs,
            quad_coeffs,
            constant,
        }
    }

    pub fn lin_coeffs(&self) -> &Vec<(VarId, E)> {
        &self.lin_coeffs
    }

    pub fn quad_coeffs(&self) -> &Vec<((VarId, VarId), E)> {
        &self.quad_coeffs
    }

    pub fn constant(&self) -> E {
        self.constant
    }

    pub fn set_constant(&mut self, constant: E) {
        self.constant = constant;
    }

    pub fn sort(&mut self) {
        self.lin_coeffs.sort_by_key(|(var_id, _)| var_id.0);
        self.quad_coeffs
            .sort_by_key(|((var_id1, var_id2), _)| (var_id1.0, var_id2.0));
    }
}

impl Default for QuadExpr {
    fn default() -> Self {
        Self {
            lin_coeffs: Vec::new(),
            quad_coeffs: Vec::new(),
            constant: E::from(0.),
        }
    }
}

impl From<E> for QuadExpr {
    fn from(constant: E) -> Self {
        Self {
            lin_coeffs: Vec::new(),
            quad_coeffs: Vec::new(),
            constant,
        }
    }
}

impl From<VarId> for QuadExpr {
    fn from(var: VarId) -> Self {
        QuadExpr {
            lin_coeffs: vec![(var, E::from(1.))],
            quad_coeffs: Vec::new(),
            constant: E::from(0.),
        }
    }
}

impl From<&VarId> for QuadExpr {
    fn from(var: &VarId) -> Self {
        QuadExpr {
            lin_coeffs: vec![(var.clone(), E::from(1.))],
            quad_coeffs: Vec::new(),
            constant: E::from(0.),
        }
    }
}

impl From<&LinExpr> for QuadExpr {
    fn from(lin_expr: &LinExpr) -> Self {
        QuadExpr {
            lin_coeffs: lin_expr.coeffs().clone(),
            quad_coeffs: Vec::new(),
            constant: lin_expr.constant(),
        }
    }
}

impl Mul<VarId> for VarId {
    type Output = QuadExpr;

    fn mul(self, rhs: VarId) -> Self::Output {
        let mut quad_expr = QuadExpr::default();
        quad_expr.quad_coeffs.push(((self, rhs), E::from(1.)));
        quad_expr
    }
}

impl Add for QuadExpr {
    type Output = QuadExpr;

    fn add(self, rhs: Self) -> Self::Output {
        let mut lin_coeffs_map: HashMap<VarId, E> = self.lin_coeffs.into_iter().collect();
        for (idx, coeff) in rhs.lin_coeffs.into_iter() {
            *lin_coeffs_map.entry(idx).or_insert(E::from(0.)) += coeff;
        }
        let mut quad_coeffs_map: HashMap<(VarId, VarId), E> =
            self.quad_coeffs.into_iter().collect();
        for (idx_pair, coeff) in rhs.quad_coeffs.into_iter() {
            *quad_coeffs_map.entry(idx_pair).or_insert(E::from(0.)) += coeff;
        }
        QuadExpr {
            lin_coeffs: lin_coeffs_map.into_iter().collect(),
            quad_coeffs: quad_coeffs_map.into_iter().collect(),
            constant: self.constant + rhs.constant,
        }
    }
}

impl AddAssign for QuadExpr {
    fn add_assign(&mut self, rhs: Self) {
        let mut lin_coeffs_map: HashMap<VarId, E> = self.lin_coeffs.iter().cloned().collect();
        for (idx, coeff) in rhs.lin_coeffs.into_iter() {
            *lin_coeffs_map.entry(idx).or_insert(E::from(0.)) += coeff;
        }
        self.lin_coeffs = lin_coeffs_map.into_iter().collect();

        let mut quad_coeffs_map: HashMap<(VarId, VarId), E> =
            self.quad_coeffs.iter().cloned().collect();
        for (idx_pair, coeff) in rhs.quad_coeffs.into_iter() {
            *quad_coeffs_map.entry(idx_pair).or_insert(E::from(0.)) += coeff;
        }
        self.quad_coeffs = quad_coeffs_map.into_iter().collect();

        self.constant += rhs.constant;
    }
}

impl Sub for QuadExpr {
    type Output = QuadExpr;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (rhs * E::from(-1.))
    }
}

impl SubAssign for QuadExpr {
    fn sub_assign(&mut self, rhs: Self) {
        *self += rhs * E::from(-1.);
    }
}

impl Mul<E> for QuadExpr {
    type Output = QuadExpr;

    fn mul(self, rhs: E) -> Self::Output {
        QuadExpr {
            lin_coeffs: self
                .lin_coeffs
                .into_iter()
                .map(|(idx, coeff)| (idx, coeff * rhs))
                .collect(),
            quad_coeffs: self
                .quad_coeffs
                .into_iter()
                .map(|(idx_pair, coeff)| (idx_pair, coeff * rhs))
                .collect(),
            constant: self.constant * rhs,
        }
    }
}

impl Mul<VarId> for LinExpr {
    type Output = QuadExpr;

    fn mul(self, rhs: VarId) -> Self::Output {
        let mut quad_expr = QuadExpr::from(&self);
        quad_expr.lin_coeffs.push((rhs, E::from(1.)));
        quad_expr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::model::LinearModel;

    #[test]
    fn test_lin_expr_add() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let var2 = model.add_var(None, -5., 5.);
        let mut expr1 = LinExpr::from(var1) + LinExpr::from(var2);
        expr1.sort();
        assert_eq!(expr1.coeffs(), &vec![(VarId(0), 1.), (VarId(1), 1.)]);
        assert_eq!(expr1.constant(), 0.);
    }

    #[test]
    fn test_lin_expr_add_assign() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let var2 = model.add_var(None, -5., 5.);
        let mut expr1 = LinExpr::from(var1);
        expr1 += LinExpr::from(var2);
        expr1.sort();
        assert_eq!(expr1.coeffs(), &vec![(VarId(0), 1.), (VarId(1), 1.)]);
        assert_eq!(expr1.constant(), 0.);
    }

    #[test]
    fn test_lin_expr_mul() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let expr1 = LinExpr::from(var1) * 2.;
        assert_eq!(expr1.coeffs(), &vec![(VarId(0), 2.)]);
        assert_eq!(expr1.constant(), 0.);
    }

    #[test]
    fn test_quad_expr_add() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let var2 = model.add_var(None, -5., 5.);
        let mut expr1 = QuadExpr::from(&var1) + QuadExpr::from(&var2);
        expr1.sort();
        assert_eq!(expr1.lin_coeffs(), &vec![(VarId(0), 1.), (VarId(1), 1.)]);
        assert_eq!(expr1.quad_coeffs(), &vec![]);
        assert_eq!(expr1.constant(), 0.);
    }

    #[test]
    fn test_quad_expr_add_assign() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let var2 = model.add_var(None, -5., 5.);
        let mut expr1 = QuadExpr::from(&var1);
        expr1 += QuadExpr::from(&var2);
        expr1.sort();
        assert_eq!(expr1.lin_coeffs(), &vec![(VarId(0), 1.), (VarId(1), 1.)]);
        assert_eq!(expr1.quad_coeffs(), &vec![]);
        assert_eq!(expr1.constant(), 0.);
    }
    #[test]
    fn test_quad_expr_mul() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let expr1 = QuadExpr::from(&var1) * 2.;
        assert_eq!(expr1.lin_coeffs(), &vec![(VarId(0), 2.)]);
        assert_eq!(expr1.quad_coeffs(), &vec![]);
        assert_eq!(expr1.constant(), 0.);
    }

    #[test]
    fn test_lin_expr_mul_var() {
        let mut model = LinearModel::new();
        let var1 = model.add_var(None, 0., 10.);
        let expr1: QuadExpr = var1 * var1;
        assert_eq!(expr1.lin_coeffs(), &vec![]);
        assert_eq!(expr1.quad_coeffs(), &vec![((VarId(0), VarId(0)), 1.)]);
        assert_eq!(expr1.constant(), 0.);
    }
}
