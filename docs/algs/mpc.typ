#import "preamble.typ": scr, underset

== Mehrotra Predictor-Corrector

The Mehrotra Predictor-Corrector is an example of a primal-dual interior point method for solving linear programs. While it does not guarantee polynomial-time convergence, in practive it typically demonstrates convergence in a small number of iterations.

The theory behind this algorithm follows from the general primal-dual interior point framework, which is described in a later section (TODO: link). For the purposes of this discussion, we restrict attention to linear programming and the application of Mehrotra's method to this problem class.

The KKT conditions @eq.lp.kkt for the linear program @eq.lp.primal can be rearranged to give the following system of equations:
$
  & bold(A) bold(x) - bold(b) = 0, \
  & bold(c) - bold(A)^T bold(y) - underline(bold(z)) - overline(bold(z)) = 0, \
  & (bold(X) - bold(L)) underline(bold(Z)) bold(e) = sigma mu bold(e), \
  & (bold(X) - bold(U)) overline(bold(Z)) bold(e) = sigma mu bold(e), \
$
where $bold(e)$ is the vector of all ones, and the matrices $bold(X)$, $bold(L)$, $bold(U)$, $underline(bold(Z))$, and $overline(bold(Z))$ are diagonal matrices with the elements of the corresponding vectors on their diagonals. The parameter $mu$ is used to ensure positivity of complimentary slackness conditions. It can be seen that this is equivalent to @eq.lp.kkt for $mu = 0$.

The Mehrotra Predictor-Corrector algorithm aims to solve a linearization of the above system, while simultaneosly driving $mu$ to zero. The linearization is given by:
$
  & bold(A) (bold(x) + Delta bold(x)) - bold(b) = 0, \
  & bold(c) - bold(A)^T (bold(y) + Delta bold(y)) - (underline(bold(z)) + Delta underline(bold(z))) - (overline(bold(z)) + Delta overline(bold(z))) = 0, \
  & (bold(X) - bold(L)) underline(bold(Z)) bold(e) + (bold(X) - bold(L)) Delta underline(bold(z)) + underline(bold(Z) Delta bold(x)) = sigma mu bold(e), \
  & (bold(X) - bold(U)) overline(bold(Z)) bold(e) + (bold(X) - bold(U)) Delta overline(bold(z)) + overline(bold(Z) Delta bold(x)) = sigma mu bold(e). \
$
Collecting constant terms on the right-hand side, this yields the following system of equations in the search directions $Delta bold(x)$, $Delta bold(y)$, $Delta underline(bold(z))$, and $Delta overline(bold(z))$:
$
  & bold(A) Delta bold(x) = bold(b) - bold(A) bold(x), \
  & bold(A)^T Delta bold(y) + Delta underline(bold(z)) + Delta overline(bold(z)) = bold(c) - bold(A)^T bold(y) - underline(bold(z)) - overline(bold(z)), \
  & (bold(X) - bold(L)) Delta underline(bold(z)) + underline(bold(Z)) Delta bold(x) = sigma mu bold(e) - (bold(X) - bold(L)) underline(bold(Z)) bold(e), \
  & (bold(X) - bold(U)) Delta overline(bold(z)) + overline(bold(Z)) Delta bold(x) = sigma mu bold(e) - (bold(X) - bold(U)) overline(bold(Z)) bold(e). \
$
The algorithm proceeds by solving the above linear system for the search directions $Delta bold(x)$, $Delta bold(y)$, $Delta underline(bold(z))$, and $Delta overline(bold(z))$. The step sizes are then computed to ensure that the updated variables remain feasible with respect to the bound constraints. Finally, the variables are updated using the computed step sizes and search directions.

The key innovation of Mehrotra's method is the use of a predictor-corrector approach to improve convergence. In the predictor step, the algorithm solves the linear system with $sigma = 0$ to obtain an initial estimate of the search directions. The value of $mu$ is then updated based on the predicted step, and the center-corrector step is performed by solving the linear system again with the updated value of $sigma$. (TODO: update with information for centering and corrector description)

Suppose the above system results in a search direction $Delta bold(x)_p$, $Delta bold(y)_p$, $Delta underline(bold(z))_p$, and $Delta overline(bold(z))_p$ in the predictor step. The algorithm must then ensure that the subsequent iteration remains feasible with respect to the bound constraints. We thus require step sizes $alpha_p$ and $alpha_d$ that ensure positivity. Taking the primal lower bound as an example, we require:
$
  & bold(x) + alpha_p Delta bold(x) >= underline(bold(x))
$
In the case that $Delta bold(x)_{p,i} >= 0$, we know that $bold(x) >= underline(bold(x))$ and $alpha_p$ = 0, so the above condition is always satisfied. Thus, we must only consider the case where $Delta bold(x)_{p,i} < 0$. Rearranging the above condition gives:
$
  alpha_p <= (underline(bold(x))_i - bold(x)_i) / (Delta bold(x)_i) #h(1em) forall i, Delta bold(x)_i < 0.
$
We ensure this condition by ensuring that $alpha_p$ is less than or equal to the minimum of the right-hand side over all $i$ where $Delta bold(x)_i < 0$. To provide some margin, we typically scale this value by a factor $tau$ in $(0, 1)$. A similar process can be used to identify the dual step size $alpha_d$.

The system of equations in @eq.mpc.system can be reduced to variables in $Delta bold(x), Delta bold(y)$, by noting that $Delta underline(bold(z))$ and $Delta bold(overline(bold(z)))$ can be expressed in terms of $Delta bold(x)$. Specifically, we have:
$
  Delta underline(bold(z)) &= (bold(X) - bold(L))^(-1) (sigma mu bold(e) - (bold(X) - bold(L)) underline(bold(Z)) bold(e) - underline(bold(Z)) Delta bold(x)), \
  &= (bold(X) - bold(L))^(-1) sigma mu bold(e) - underline(bold(z)) - (bold(X) - bold(L))^(-1) underline(bold(Z)) Delta bold(x), \
  Delta overline(bold(z)) &= (bold(X) - bold(U))^(-1) (sigma mu bold(e) - (bold(X) - bold(U)) overline(bold(Z)) bold(e) - overline(bold(Z)) Delta bold(x)), \
  &= (bold(X) - bold(U))^(-1) sigma mu bold(e) - overline(bold(z)) - (bold(X) - bold(U))^(-1) overline(bold(Z)) Delta bold(x)).
$ <eq.mpc.zdx>
The system @eq.mpc.system can then be reduced to the following system in $Delta bold(x)$ and $Delta bold(y)$:
$
  & bold(A) Delta bold(x) = bold(b) - bold(A) bold(x), \
  & -((bold(X) - bold(L))^(-1) underline(bold(Z)) + (bold(X) - bold(U))^(-1) overline(bold(Z))) Delta bold(x) + bold(A)^T Delta bold(y) \
  & #h(2em) = bold(c) - bold(A)^T bold(y) - underline(bold(z)) - overline(bold(z)) + underline(bold(z)) + overline(bold(z)) - sigma mu (bold(X) - bold(L))^(-1) bold(e) - sigma mu (bold(X) - bold(U))^(-1) bold(e), \
  & #h(2em) = bold(c) - bold(A)^T bold(y) - sigma mu ( (bold(X) - bold(L))^(-1) +(bold(X) - bold(U))^(-1) ) bold(e),
$
and $Delta underline(bold(z))$ and $Delta overline(bold(z))$ are as defined in @eq.mpc.zdx. Note that this formulation retains the sparsity of the original problem. This allows the usage of sparse linear system solvers, such as Cholesky factorization to efficiently solve the system of equations.

// === Implementing Mehrotra's Predictor-Corrector
