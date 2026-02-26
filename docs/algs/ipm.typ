#import "preamble.typ": scr, underset

== Interior Point Method

We consider the problem in @eq.nlp.general. The interior point method is a general framework for solving nonlinear optimization problems. It is based on the idea of iteratively solving a sequence of approximations to the original problem, where each approximation is obtained by adding a barrier term to the objective function that penalizes infeasibility with respect to the constraints. The barrier term is typically chosen to be a logarithmic function of the distance to the boundary of the feasible region, which ensures that the iterates remain strictly feasible. The method proceeds by solving a sequence of barrier problems, where the barrier parameter is gradually reduced to zero, until convergence to a solution of the original problem is achieved. The interior point method can be applied to a wide range of optimization problems, including linear programming, quadratic programming, and nonlinear programming. In the context of nonlinear programming, the interior point method is often implemented using a primal-dual approach, where both the primal and dual variables are updated at each iteration.

Without loss of generality, we will focus on the case of nonlinear programming problems with only equality constraints, as the presence of inequality constraints can be easily handled by introducing slack variables. Additionally, we assume that $bold(x)$ lies in the set $[bold(l), bold(u)]$. This results in the problem,
$
  & underset(#[minimize], bold(x)) & #h(2mm) & f(bold(x)), \
  & #[subject to],                 &         & bold(g)(bold(x)) = 0,          && : bold(y), \
  &                                &         & bold(x) in [bold(l), bold(u)], && : underline(bold(z)), overline(bold(z)), \
$ <eq.ipm.primal>
where we associated dual variables $bold(y)$, $underline(bold(z))$, and $overline(bold(z))$ with the equality constraint, lower bound, and upper bound, respectively. The KKT conditions for this problem are given by:
$
  & nabla f(bold(x)) - nabla bold(g)(bold(x))^T bold(y) - underline(bold(z)) - overline(bold(z)) = 0, \
  & bold(g)(bold(x)) = 0, \
  & underline(bold(z))^T (bold(x) - bold(l)) = 0, overline(bold(z))^T (bold(x) - bold(u)) = 0, \
  & bold(x) in [bold(l), bold(u)], underline(bold(z)) >= 0, overline(bold(z)) <= 0. \
$
where the first equation represents stationarity (dual feasibility), the second equation represents primal feasibility, the third equation represents complementary slackness, and the last two equations represent primal-dual variable feasibility. It is these set of equations we aim to solve. In practice this is difficult to solve, due to the nonlinearity of the stationarity condition and the complementarity conditions. The interior point method addresses this by solving a sequence of approximations to the original problem, where each approximation is obtained by adding a barrier term to the objective function that penalizes infeasibility with respect to the constraints. The barrier term is typically chosen to be a logarithmic function of the distance to the boundary of the feasible region, which ensures that the iterates remain strictly feasible. This is handled by introducing a barrier parameter $mu$ that is gradually reduced to zero, until convergence to a solution of the original problem is achieved, solving the problem,
$
  & underset(#[minimize], bold(x)) & #h(2mm) & f(bold(x)) - mu sum_{i=1}^n (log(bold(x)_i - bold(l)_i) + log(bold(u)_i - bold(x)_i)), \
  & #[subject to], & & bold(g)(bold(x)) = 0, && : bold(y), \
$
This results in the KKT conditions,
$
  & nabla f(bold(x)) - nabla bold(g)(bold(x))^T bold(y) - underline(bold(z)) - overline(bold(z)) = 0, \
  & bold(g)(bold(x)) = 0, \
  & underline(bold(z))^T (bold(x) - bold(l)) = mu bold(e), overline(bold(z))^T (bold(x) - bold(u)) = mu bold(e), \
  & bold(x) in [bold(l), bold(u)], underline(bold(z)) >= 0, overline(bold(z)) <= 0, \
$ <eq.ipm.kkt>
where $n$ is the number of decision variables. As $mu$ approaches zero, the above KKT conditions approach the original KKT conditions for the problem in @eq.ipm.primal. The introduction of the variable $mu$ ensures that the iterates remain strictly feasible with respect to the bound constraints, while also providing a mechanism for driving the iterates towards optimality by gradually reducing $mu$ to zero.

Supposing the iterates respect the bound constraints, we can express the complementarity conditions as:
$
  & (bold(X) - bold(L)) underline(bold(Z)) bold(e) = mu bold(e), \
  & (bold(X) - bold(U)) overline(bold(Z)) bold(e) = mu bold(e), \
$ <eq.ipm.cs>
where the matrices $bold(X)$, $bold(L)$, $bold(U)$, $underline(bold(Z))$, and $overline(bold(Z))$ are diagonal matrices with the elements of the corresponding vectors on their diagonals. The interior point method aims to solve the KKT conditions speciied in @eq.ipm.kkt and @eq.ipm.cs by solving a sequence of approximations to the original problem with diminishing $mu$. The linearization of these equations is given by
$
  & nabla f(bold(x)) + nabla^2 f(bold(x)) Delta bold(x) - bold(y) nabla^2 bold(g)(bold(x)) Delta bold(x) - nabla bold(g)(bold(x))^T (bold(y) + Delta bold(y)) - (underline(bold(z)) + Delta underline(bold(z))) - (overline(bold(z)) + Delta overline(bold(z))) = 0, \
  & bold(g)(bold(x)) + nabla bold(g)(bold(x)) Delta bold(x) = 0, \
  & (bold(X) - bold(L)) underline(bold(Z)) bold(e) + (bold(X) - bold(L)) Delta underline(bold(z)) + underline(bold(Z)) Delta bold(x) = mu bold(e), \
  & (bold(X) - bold(U)) overline(bold(Z)) bold(e) + (bold(X) - bold(U)) Delta overline(bold(z)) + overline(bold(Z)) Delta bold(x) = mu bold(e). \
$
Combining the differeces on the left and the non-differential terms on the right, we have
$
  & nabla^2 f(bold(x)) Delta bold(x) - bold(y)^2 nabla^2 bold(g)(bold(x)) Delta bold(x) - nabla bold(g)(bold(x))^T Delta bold(y) - Delta underline(bold(z)) - Delta overline(bold(z)) = - (nabla f(bold(x)) - nabla bold(g)(bold(x))^T bold(y) - underline(bold(z)) - overline(bold(z))), \
  & nabla bold(g)(bold(x)) Delta bold(x) = -bold(g)(bold(x)), \
  & (bold(X) - bold(L)) Delta underline(bold(z)) + underline(bold(Z)) Delta bold(x) = mu bold(e) - (bold(X) - bold(L)) underline(bold(Z)) bold(e), \
  & (bold(X) - bold(U)) Delta overline(bold(z)) + overline(bold(Z)) Delta bold(x) = mu bold(e) - (bold(X) - bold(U)) overline(bold(Z)) bold(e). \
$
Unfortunately, the above system is not symetric, making it more difficult to solve. However, we can reduce the system to a symmetric system in $Delta bold(x)$ and $Delta bold(y)$ by expressing $Delta underline(bold(z))$ and $Delta overline(bold(z))$ in terms of $Delta bold(x)$, by recognizing
$
  & Delta underline(bold(z)) = (bold(X) - bold(L))^(-1) (mu bold(e) - (bold(X) - bold(L)) underline(bold(Z)) bold(e) - underline(bold(Z)) Delta bold(x)), \
  & Delta overline(bold(z)) = (bold(X) - bold(U))^(-1) (mu bold(e) - (bold(X) - bold(U)) overline(bold(Z)) bold(e) - overline(bold(Z)) Delta bold(x)). \
$
Combining the above with the first two equations gives the following system in $Delta bold(x)$ and $Delta bold(y)$:
$
  & (nabla^2 f(bold(x)) - nabla^2 bold(g)^T (bold(x)) bold(y) + (bold(X) - bold(L))^(-1) underline(bold(Z)) + (bold(X) - bold(U))^(-1) overline(bold(Z))) Delta bold(x) - nabla bold(g)(bold(x))^T Delta bold(y) \
  & #h(2em) = - (nabla f(bold(x)) - nabla bold(g)(bold(x))^T bold(y) - underline(bold(z)) - overline(bold(z))) - underline(bold(z)) - overline(bold(z)) + mu ( (bold(X) - bold(L))^(-1) + (bold(X) - bold(U))^(-1) ) bold(e), \
  & #h(2em) = -nabla f(bold(x)) + nabla bold(g)(bold(x))^T bold(y) + underline(bold(z)) + overline(bold(z)) - underline(bold(z)) - overline(bold(z)) + mu ( (bold(X) - bold(L))^(-1) + (bold(X) - bold(U))^(-1) ) bold(e), \
  & #h(2em) = -nabla f(bold(x)) + nabla bold(g)(bold(x))^T bold(y) + mu ( (bold(X) - bold(L))^(-1) + (bold(X) - bold(U))^(-1) ) bold(e), \
  & -nabla bold(g)(bold(x)) Delta bold(x) = bold(g)(bold(x)). \
$
This system is symmetric and can be solved using efficient linear solvers, such as Cholesky factorization. The solution to this system provides the search directions for the primal and dual variables.

=== On the application of interior point methods

In practical applications of interior point methods, the simplications above are, perhaps surprisngly, inefficient. This is due to the fact that some of the variables in the system are computed for other purposes for optimization. In particular, the residual of the KKT conditions is computed at each iteration to check for convergence, and the values of $underline(bold(z))$ and $overline(bold(z))$ are used to compute the barrier term in the objective function. As such, it is more efficient to compute the search directions for $Delta underline(bold(z))$ and $Delta overline(bold(z))$ directly from the residuals, rather than expressing them in terms of $Delta bold(x)$ and solving a reduced system. This allows us to retain the sparsity of the original problem, which can be exploited by sparse linear solvers to efficiently solve the system of equations. For that purpose let us define the residual of the KKT conditions as
$
  & bold(r)_D = -nabla f(bold(x)) + nabla bold(g)(bold(x))^T bold(y) + underline(bold(z)) + overline(bold(z)), \
  & bold(r)_P = -bold(g)(bold(x)), \
  & bold(r)_underline(Z) = -(bold(X) - bold(L)) underline(bold(Z)) bold(e), \
  & bold(r)_overline(Z) = -(bold(X) - bold(U)) overline(bold(Z)) bold(e). \
$ <eq.ipm.residual>
The search directions can then be computed by solving the following system of equations:
$
  & nabla^2 f(bold(x)) Delta bold(x) - bold(y) nabla^2 bold(g)(bold(x)) Delta bold(x) - nabla bold(g)(bold(x))^T Delta bold(y) - Delta underline(bold(z)) - Delta overline(bold(z)) = bold(r)_D, \
  & -nabla bold(g)(bold(x)) Delta bold(x) = bold(r)_P, \
  & (bold(X) - bold(L)) Delta underline(bold(z)) + underline(bold(Z)) Delta bold(x) = mu bold(e) + bold(r)_underline(Z), \
  & (bold(X) - bold(U)) Delta overline(bold(z)) + overline(bold(Z)) Delta bold(x) = mu bold(e) + bold(r)_overline(Z). \
$
This system can be solved using efficient linear solvers, such as Cholesky factorization, while retaining the sparsity of the original problem. The solution to this system provides the search directions for the primal and dual variables. Similarly, the simplified system can be solved by substituting the expressions for $Delta underline(bold(z))$ and $Delta overline(bold(z))$ in terms of the residuals, which also retains the sparsity of the original problem, resulting in the system:
$
  & (nabla^2 f(bold(x)) - nabla^2 bold(g)^T (bold(x)) bold(y) + (bold(X) - bold(L))^(-1) underline(bold(Z)) + (bold(X) - bold(U))^(-1) overline(bold(Z))) Delta bold(x) - nabla bold(g)(bold(x))^T Delta bold(y) \
  & #h(2em) = bold(r)_D + mu ( (bold(X) - bold(L))^(-1) + (bold(X) - bold(U))^(-1) ) bold(e) + (bold(X) - bold(L))^(-1) bold(r)_underline(Z) + (bold(X) - bold(U))^(-1) bold(r)_overline(Z), \
  & -nabla bold(g)(bold(x)) Delta bold(x) = bold(r)_P. \
$
While such an approach may appear unnecessary, it will be shown in the next section that it can lend itself well to predictor-corrector methods, which can significantly improve the convergence of the interior point method.

=== Predictor-Corrector Methods

An approach to improve the convergence of the interior point method is to use a predictor-corrector approach, where a predictor step is first taken to obtain an estimate of the search directions, and then a corrector step is taken to refine the search directions based on the predicted step. The predictor step is typically performed by solving the linear system with $mu = 0$, which corresponds to ignoring the barrier term in the objective function. The corrector step is then performed by solving the linear system again with an updated value of $mu$ that accounts for the predicted step. This approach can lead to faster convergence and improved numerical stability compared to a standard interior point method.
