#import "preamble.typ": scr, underset

= Introduction

Copters is a library for solving optimization problems, both linear and nonlinear. In this section, we will introduce the basic concepts of optimization theory that are relevant to the algorithms implemented in copters. We will discuss the formulation of optimization problems, as well as key concepts that will be used to describe the algorithms.

We begin with a general formulation of optimization problems, and then specialize to linear programming (LP) and nonlinear programming (NLP). This general form seeks to optimize (minimize or maximize) an objective function subject to constraints on the decision variables. The general (non-linear) optimization problem can be stated as:
$
  & underset(#[minimize], bold(x)) & #h(2mm) & f(bold(x)), \
  & #[subject to],                 &         & bold(g)(bold(x)) = 0, \
  // && : y, \
  &                                &         & bold(h)(bold(x)) <= 0, \
  // && : z, \
  &                                &         & bold(x) in cal(X), \ // && : w \
$ <eq.nlp.general>
where $f$ is the objective function, $bold(g)$ are the equality constraints, $bold(h)$ are the inequality constraints, and $bold(X)$ represents the feasible set for the decision variables.

TODO: Add discussion of duality.

We associate dual variables $bold(y)$, $bold(z)$, and $bold(w)$ with the equality constraints, inequality constraints, and set membership constraints, respectively. The associated Lagrangian is given by:
$
  cal(L)(bold(x), bold(y), bold(z) >= 0, bold(w) in W) = f(bold(x)) - bold(y)^T bold(g)(bold(x)) - bold(z)^T bold(h)(bold(x)) - I_{cal(W)}(bold(x)).
$ <eq.lagrangian.general>
The problem @eq.nlp.general can be equivalently written as:
$
  underset(min, bold(x)) underset(max, (bold(y), bold(z) >= 0, bold(w) in W)) cal(L)(bold(x), bold(y), bold(z), bold(w))
$ <eq.primal.general>
where the maximization is over the dual variables.It can then be shown that we can construct the lower bound:
$
  underset(min, bold(x)) underset(max, (bold(y), bold(z) >= 0, bold(w) in cal(W))) cal(L)(bold(x), bold(y), bold(z), bold(w)) >= underset(max, (bold(y), bold(z) >= 0, bold(w) in cal(W))) underset(min, bold(x)) cal(L)(bold(x), bold(y), bold(z), bold(w)).
$ <eq.weak.duality.general>
The lower bound is itself an optimization problem and is referred to as the dual problem. The weak duality theorem states that the optimal value of the primal problem is always greater than or equal to the optimal value of the dual problem. Under certain conditions, known as strong duality conditions, the two optimal values are equal.

The dual problem can be equivalently written as:
$
  & underset(#[maximize], (bold(y), bold(z), bold(w))) & #h(2mm) & underset(min, bold(x)) cal(L)(bold(x), bold(y), bold(z), bold(w)), \
  & #[subject to], & & bold(z) >= 0, bold(w) in cal(W). \
$ <eq.dual.general>

The dual problem can be used to derive necessary and sufficient conditions for optimality of the primal problem, known as the Karush-Kuhn-Tucker (KKT) conditions, which require satisfaction of the primal feasibility, dual feasibility, complementary slackness, and stationarity conditions. The complementary slackness condition ensures that for each inequality constraint, either the constraint is active (i.e., holds with equality) or the corresponding dual variable is zero. The KKT conditions for optimality of the primal and dual problems are given by:
$
  & bold(g)(bold(x)) = 0, \
  & bold(h)(bold(x)) <= 0, \
  & nabla f(bold(x)) - nabla bold(g)(bold(x))^T bold(y) - nabla bold(h)(bold(x))^T bold(z) - nabla I_{cal(W)}(bold(x)) = 0, \
  & bold(z)^T bold(h)(bold(x)) = 0, \
  & bold(z) >= 0, bold(x) in cal(X), \
$ <eq.kkt.general>
where the first two equations represent primal feasibility, the third equation represents stationarity, the fourth equation represents complementary slackness, and the last two equations represent dual feasibility.

In the following subsections we will explore specific cases of optimization problems, namely linear programming (LP) and nonlinear programming (NLP), and derive their respective dual problems and KKT conditions. In these instances the above equations may simplify due to the specific structure of the objective functions and constraints.

== Linear Programming (LP)

The LP problems is a special case of the general optimization problem where the objective function and constraints are all linear. A standard form of a linear programming problem can be stated as:
$
  & underset(#[minimize], bold(x)) & #h(2mm) & bold(c)^T bold(x), \
  & #[subject to], & & bold(A) bold(x) = bold(b), & & : bold(y) \
  & & & underline(bold(x)) <= bold(x) <= overline(bold(x)). & #h(2mm) & : underline(bold(w)), overline(bold(w))
$ <eq.lp.primal>
Here, we have associated dual variables, $y$, $underline(w)$, and $overline(w)$ with the linear constraint, lower bound, and upper bound, respectively. In this problem the objective function is now a linear function $f(bold(x)) = bold(c)^T bold(x)$, the equality constraints are given by $bold(g)(bold(x)) = bold(A) bold(x) - bold(b)$, and the inequality constraint is ignored. It can be shown that inequality constraints can easily be reformulated as equality constraints. The set membership constraint restricts $cal(X) = \[ underline(bold(x)), overline(bold(x)) \]$.

We can then construct the associated Lagrangian
$
  scr(L)(bold(x), bold(y), underline(bold(w)) >= 0, overline(bold(w)) <= 0) = bold(c)^T bold(x) - bold(y)^T (bold(A) bold(x) - bold(b)) - underline(bold(w))^T (bold(x) - underline(bold(x))) - overline(bold(w))^T (bold(x) - overline(bold(x))).
$ <eq.lp.lagrangian>

The dual problem can then be constructed as:
$
  & underset(#[maximize], (bold(y), underline(bold(w)) >= 0, overline(bold(w)) <= 0)) & #h(2mm) & bold(b)^T bold(y) + underline(bold(w))^T underline(bold(x)) + overline(bold(w))^T overline(bold(x)), \
  & #[subject to], && A^T y + underline(z) - overline(z) = c, \
  & && underline(bold(w)) >= 0, overline(bold(w)) <= 0. \
$ <eq.lp.dual>

Necessary and sufficient conditions for optimality of the primal and dual problems are given by the KKT conditions:
$
  & bold(A) bold(x) = bold(b), \
  & bold(c) - bold(A)^T bold(y) - underline(bold(w)) + overline(bold(w)) = 0, \
  & underline(bold(w))^T (bold(x) - underline(bold(x))) = 0, overline(bold(w))^T (bold(x) - overline(bold(x))) = 0, \
  & underline(bold(x)) <= bold(x) <= overline(bold(x)), underline(bold(w)) >= 0, overline(bold(w)) <= 0, \
$ <eq.lp.kkt>
where the first two equations are the primal and dual feasibility conditions, the next three equations are the complementary slackness conditions, and the last two equations are the dual feasibility conditions.

== Nonlinear Programming (NLP)
A general nonlinear programming (NLP) problem can be stated as:
$
  & underset(#[minimize], x) & #h(2mm) & f(x), \
  & #[subject to],           &         & g(x) = 0,     && : y, \
  &                          &         & x \in [l, u], && : underline(z), overline(z) \
$ <eq.nlp>
where $f: R^n -> R$ is the objective function, $g: R^n -> R^m$ are the equality constraints, and $x in [l, u]$ represents the bound constraints on the decision variables. The associated Langrangian is given by:
$
  scr(L)(x, y, underline(z) >= 0, overline(z) <= 0) = f(x) - y^T g(x) - underline(z)^T (x - l) - overline(z)^T (x - u).
$ <eq.nlp.lagrangian>
The KKT conditions for optimality are given by:
$
  & g(x) = 0, \
  & nabla f(x) - nabla g(x)^T y - underline(z) + overline(z) = 0, \
  & underline(z)^T (x - l) = 0, overline(z)^T (x - u) = 0, \
  & x in [l, u], underline(z) >= 0, overline(z) <= 0, \
$ <eq.nlp.kkt>
where the first equation represents primal feasibility, the second equation represents dual feasibility, the next two equations represent complementary slackness, and the last three equations represent the bound constraints on the primal and dual variables.

The KKT conditions @eq.nlp.kkt are necessary for optimality under certain regularity conditions, such as the Mangasarian-Fromovitz constraint qualification (MFCQ). They are sufficient for optimality if the objective function $f$ is convex and the constraint functions $g$ are affine. They are sufficient for local optimality if $f$ and $g$ are twice continuously differentiable and the second-order sufficient conditions hold, given by the positive definiteness of the Lagrangian Hessian on the critical cone.

