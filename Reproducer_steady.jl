using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator

# using GridapODEs.ODETools: ThetaMethodLinear
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t

θ = 0.5

u(x,t) = VectorValue(x[1],x[2])
u(t::Real) = x -> u(x,t)

p(x,t) = (x[1]-x[2])
p(t::Real) = x -> p(x,t)
q(x) = t -> p(x,t)

f(t) = x -> ∂t(u)(t)(x)-Δ(u(t))(x)+ ∇(p(t))(x)
g(t) = x -> (∇⋅u(t))(x)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)
order = 2

V0 = FESpace(
  reffe=:Lagrangian, order=order, valuetype=VectorValue{2,Float64},
  conformity=:H1, model=model, dirichlet_tags="boundary")
Q = TestFESpace(
  model=model,
  order=order-1,
  reffe=:Lagrangian,
  valuetype=Float64,
  conformity=:H1,
  constraint=:zeromean)

U = TrialFESpace(V0,u(0.0))
P = TrialFESpace(Q)
X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

function res(x,y)
  u,p = x
  v,q = y
  inner(∇(u),∇(v)) - (∇⋅v)*p + q*(∇⋅u) - inner(v,f(0.0)) - q*g(0.0)
end


t_Ω_ad = FETerm(res,trian,quad)
#t_Ω_ad = FETerm(res,jac,jac_t,trian,quad)
op = FEOperator(X,Y,t_Ω_ad)

xh = solve(op)

l2(w) = w⋅w
tol = 1.0e-10
uh = xh[1]
ph = xh[2]
e = u(0.0) - uh
el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
e = p(0.0) - ph
el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
@test el2 < tol
