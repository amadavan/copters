== Interior Point Method (IPM)

The interior point method is an iterative algorithm that aims to solve general nonlinear programming problems. 


The KKT conditions in @eq.nlp.kkt are unfortunately non-convex posing a significant challenge for optimization. Instead of trying to directly solve for the solution of the KKT system, the interior point method iteratively optimizes over a linearization of the KKT conditions. Consider the linearization about the point $(bold(x)_0, bold(y)_0, underline(bold(z))_0, overline(bold(z))_0)$,
$
  & bold(g)(bold(x)_0) + nabla bold(g)(bold(x)_0) Delta bold(x)  = 0, \
  & (nabla f(bold(x)_0) + nabla^2 f(bold(x)_0) Delta bold(x)) - (nabla bold(g)(bold(x_0))^T bold(y)_0 + bold(y)_0^T nabla^2 bold(g)(bold(x_0)) Delta bold(x) + nabla bold(g)(bold(x)_0)^T Delta bold(y)) \
  & #h(2em) - (underline(bold(z))_0 + Delta underline(bold(z))) - (overline(bold(z))_0 + Delta overline(bold(z))) = 0, \
  & underline(bold(Z))_0 (bold(X)_0 - bold(L)) bold(e) + underline(bold(Z))_0 Delta bold(x) + (bold(X)_0 - bold(L)) Delta underline(bold(z)) = 0, \
  & overline(bold(Z))_0 (bold(X)_0 - bold(U)) bold(e) + overline(bold(Z))_0 Delta bold(x) + (bold(X)_0 - bold(U)) Delta overline(bold(z)) = 0, \
  & bold(x) in [bold(l), bold(u)], underline(bold(z)) >= 0, overline(bold(z)) <= 0, \
$ <eq.ipm.kkt>
where $bold(X), underline(bold(Z)), overline(bold(Z)), bold(L), bold(U)$ are representations of their corresponding vectors as diagonal matrices. The algorithm will ignore the bounding constraints, instead choosing to restrict step sizes in order to ensure their feasibility. Thus, we require the diagonalized form to ensure satisfaction of the constraints, as the transpose variant may not produce a feasible solution. Collecting like the steps on the left and the constant values on the right, we have
$
  & nabla bold(g)(bold(x)_0) Delta bold(x) = -bold(g)(bold(x)_0), \
  & (nabla^2 f(bold(x)_0) - bold(y)_0^T nabla^2 bold(g)(bold(x)_0)) Delta bold(x) - nabla bold(g)(bold(x)_0)^T Delta bold(y) - Delta underline(bold(z)) - Delta overline(bold(z)) \
  & #h(2em) = -f(bold(x)_0) + nabla bold(g)(bold(x)_0)^T bold(y)_0 + underline(bold(z))_0 + overline(bold(z))_0, \
  & underline(bold(Z))_0 Delta bold(x) + (bold(X)_0 - bold(L)) Delta underline(bold(z)) = - underline(bold(Z))_0 (bold(X)_0 - bold(L)) bold(e), \
  & overline(bold(Z))_0 Delta bold(x) + (bold(X)_0 - bold(U)) Delta overline(bold(z)) = - overline(bold(Z))_0 (bold(X)_0 - bold(U)) bold(e),
$ <eq.ipm.system>

=== On the implementation of the interior point method

Computing an iteration of the IPM requires solving the linear system in @eq.ipm.system. It can be seen that the updates for $Delta underline(bold(z)), Delta overline(bold(z))$ can be expressed as functions of $Delta bold(x)$. Specifically,
$
  & Delta underline(bold(z)) = (bold(X)_0 - L)^(-1) ( -underline(bold(Z))_0(bold(X)_0 - bold(L)) bold(e) - underline(bold(Z))_0 Delta bold(x)), \
  & Delta overline(bold(z)) = (bold(X)_0 - U)^(-1) ( -overline(bold(Z))_0(bold(X)_0 - bold(U)) bold(e) - overline(bold(Z))_0 Delta bold(x)). \
$
This can be further simplified by noting that matrix multiplication of diagonal matrices is commutative, which allows us to express the updates as
$
  & Delta underline(bold(z)) = -underline(bold(z))_0 - (bold(X)_0 - L)^(-1) underline(bold(Z))_0 Delta bold(x), \
  & Delta overline(bold(z)) =  -overline(bold(z))_0 - (bold(X)_0 - U)^(-1) overline(bold(Z))_0 Delta bold(x). \
$
