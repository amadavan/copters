use crate::E;

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Const(E),
    Var(Box<Var>),
    Param(Box<Param>),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, i32),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
}

impl Expr {
    pub fn evaluate(&self, var_values: &[E]) -> E {
        match self {
            Expr::Const(c) => *c,
            Expr::Var(v) => v.value,
            Expr::Param(p) => p.value,
            Expr::Add(e1, e2) => e1.evaluate(var_values) + e2.evaluate(var_values),
            Expr::Sub(e1, e2) => e1.evaluate(var_values) - e2.evaluate(var_values),
            Expr::Mul(e1, e2) => e1.evaluate(var_values) * e2.evaluate(var_values),
            Expr::Div(e1, e2) => e1.evaluate(var_values) / e2.evaluate(var_values),
            Expr::Pow(base, exp) => base.evaluate(var_values).powi(*exp),
            Expr::Sin(arg) => arg.evaluate(var_values).sin(),
            Expr::Cos(arg) => arg.evaluate(var_values).cos(),
        }
    }

    pub fn simplify(&self) -> Expr {
        match self {
            Expr::Const(c) => match c {
                0.0 => Expr::Const(0.0),
                1.0 => Expr::Const(1.0),
                _ => self.clone(),
            },
            Expr::Add(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (Expr::Const(c1), Expr::Const(c2)) => Expr::Const(c1 + c2),
                    (Expr::Const(0.0), _) => se2,
                    (_, Expr::Const(0.0)) => se1,
                    _ => Expr::Add(Box::new(se1), Box::new(se2)),
                }
            }
            Expr::Sub(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (Expr::Const(c1), Expr::Const(c2)) => Expr::Const(c1 - c2),
                    (_, Expr::Const(0.0)) => se1,
                    _ => Expr::Sub(Box::new(se1), Box::new(se2)),
                }
            }
            Expr::Mul(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (Expr::Const(c1), Expr::Const(c2)) => Expr::Const(c1 * c2),
                    (Expr::Const(0.0), _) | (_, Expr::Const(0.0)) => Expr::Const(0.0),
                    (Expr::Const(1.0), _) => se2,
                    (_, Expr::Const(1.0)) => se1,
                    _ => Expr::Mul(Box::new(se1), Box::new(se2)),
                }
            }
            Expr::Div(e1, e2) => {
                let se1 = e1.simplify();
                let se2 = e2.simplify();
                match (&se1, &se2) {
                    (Expr::Const(c1), Expr::Const(c2)) if *c2 != 0.0 => Expr::Const(c1 / c2),
                    (Expr::Const(0.0), _) => Expr::Const(0.0),
                    (_, Expr::Const(1.0)) => se1,
                    _ => Expr::Div(Box::new(se1), Box::new(se2)),
                }
            }
            Expr::Pow(base, exp) => {
                let sbase = base.simplify();
                match &sbase {
                    Expr::Const(c) => Expr::Const(c.powi(*exp)),
                    _ => Expr::Pow(Box::new(sbase), *exp),
                }
            }
            Expr::Sin(arg) => {
                let sarg = arg.simplify();
                match &sarg {
                    Expr::Const(c) => Expr::Const(c.sin()),
                    _ => Expr::Sin(Box::new(sarg)),
                }
            }
            Expr::Cos(arg) => {
                let sarg = arg.simplify();
                match &sarg {
                    Expr::Const(c) => Expr::Const(c.cos()),
                    _ => Expr::Cos(Box::new(sarg)),
                }
            }
            _ => self.clone(),
        }
    }

    pub fn diff(&self, var: &Box<Var>) -> Expr {
        match self {
            Expr::Const(_) => Expr::Const(0.0),
            Expr::Var(var_) => {
                if var == var_ {
                    Expr::Const(1.0)
                } else {
                    Expr::Const(0.0)
                }
            }
            Expr::Param(param) => Expr::Const(0.0),
            Expr::Add(e1, e2) => Expr::Add(Box::new(e1.diff(var)), Box::new(e2.diff(var))),
            Expr::Sub(e1, e2) => Expr::Sub(Box::new(e1.diff(var)), Box::new(e2.diff(var))),
            Expr::Mul(e1, e2) => Expr::Add(
                Box::new(Expr::Mul(Box::new(e1.diff(var)), e2.clone())),
                Box::new(Expr::Mul(e1.clone(), Box::new(e2.diff(var)))),
            ),
            Expr::Div(e1, e2) => Expr::Div(
                Box::new(Expr::Sub(
                    Box::new(Expr::Mul(Box::new(e1.diff(var)), e2.clone())),
                    Box::new(Expr::Mul(e1.clone(), Box::new(e2.diff(var)))),
                )),
                Box::new(Expr::Mul(e2.clone(), e2.clone())),
            ),
            Expr::Pow(base, exp) => {
                if *exp == 0 {
                    Expr::Const(0.0)
                } else {
                    let new_exp = *exp - 1;
                    let coeff = *exp as f64;
                    Expr::Mul(
                        Box::new(Expr::Mul(
                            Box::new(Expr::Const(coeff)),
                            Box::new(Expr::Pow(base.clone(), new_exp)),
                        )),
                        Box::new(base.diff(var)),
                    )
                }
            }
            Expr::Sin(arg) => Expr::Mul(Box::new(Expr::Cos(arg.clone())), Box::new(arg.diff(var))),
            Expr::Cos(arg) => Expr::Mul(
                Box::new(Expr::Mul(
                    Box::new(Expr::Const(-1.0)),
                    Box::new(Expr::Sin(arg.clone())),
                )),
                Box::new(arg.diff(var)),
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Var {
    name: String,
    value: E,
    rc: E,
}

impl Var {
    pub fn new(name: String) -> Self {
        Self {
            name,
            value: E::default(),
            rc: E::default(),
        }
    }

    pub fn as_mut(&mut self) -> &mut Self {
        self
    }

    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn get_value(&self) -> E {
        self.value
    }

    pub fn get_rc(&self) -> E {
        self.rc
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    name: String,
    value: E,
}

impl Param {
    pub fn get_name(&self) -> &str {
        &self.name
    }

    pub fn as_mut(&mut self) -> &mut Self {
        self
    }

    pub fn get_value(&self) -> E {
        self.value
    }

    pub fn set_value(&mut self, value: E) {
        self.value = value;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sense {
    Equal,
    LessEqual,
    GreaterEqual,
}

impl From<&str> for Sense {
    fn from(s: &str) -> Self {
        match s {
            "==" | "=" => Sense::Equal,
            "<=" | "<" => Sense::LessEqual,
            ">=" | ">" => Sense::GreaterEqual,
            _ => panic!("Invalid sense string: {}", s),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constr {
    lhs: Expr,
    rhs: Expr,
    sense: Sense, // "==" or "<=" or ">="
    dual: E,
}

impl Constr {
    pub fn new(lhs: Expr, rhs: Expr, sense: Sense) -> Self {
        Self {
            lhs,
            rhs,
            sense,
            dual: E::default(),
        }
    }

    pub fn as_mut(&mut self) -> &mut Self {
        self
    }

    pub fn get_lhs(&self) -> &Expr {
        &self.lhs
    }

    pub fn get_rhs(&self) -> &Expr {
        &self.rhs
    }

    pub fn get_sense(&self) -> &Sense {
        &self.sense
    }

    pub fn get_dual(&self) -> E {
        self.dual
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    vars: Vec<Var>,
    constrs: Vec<Constr>,
    obj: Expr,
}

impl Model {
    pub fn get_vars(&self) -> &[Var] {
        &self.vars
    }

    pub fn get_constrs(&self) -> &[Constr] {
        &self.constrs
    }

    pub fn get_obj(&self) -> &Expr {
        &self.obj
    }

    pub fn get_obj_mut(&mut self) -> &mut Expr {
        &mut self.obj
    }

    pub fn add_variable(&mut self, var: Var) -> &Var {
        let idx = self.vars.len();
        self.vars.push(var);
        &self.vars[idx]
    }

    pub fn add_constraint(&mut self, constr: Constr) -> &Constr {
        let idx = self.constrs.len();
        self.constrs.push(constr);
        &self.constrs[idx]
    }

    pub fn set_objective(&mut self, obj: Expr) {
        self.obj = obj;
    }
}
