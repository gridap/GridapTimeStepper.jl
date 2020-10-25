using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces #: get_algebraic_operator, get_cell_basis, get_cell_jacobian
using BlockArrays

u(x,t) = VectorValue(x[1],x[2])*t
u(t::Real) = x -> u(x,t)
p(x,t) = (x[1]-x[2])*t
p(t::Real) = x -> p(x,t)

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

U = TransientTrialFESpace(V0,u)
P = TrialFESpace(Q)
X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

function res(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  inner(ut,v) + q*(∇⋅u)
end

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)
ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

x_t = FEFunction(X0,rand(num_free_dofs(X0)))
x = FEFunction(X0,rand(num_free_dofs(X0)))
dy = get_cell_basis(Y)
dx = get_cell_basis(X0)
t1 = FETerm( (x,dy) -> res(0.0,x,x_t,dy), trian, quad )
get_cell_jacobian(t1,x,dx,dy)
