module StokesEquationWithAutoDiffTests

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using Gridap.FESpaces #: get_algebraic_operator, get_cell_basis, get_cell_jacobian
using BlockArrays

# using GridapODEs.ODETools: ThetaMethodLinear
import Gridap: ∇
import GridapODEs.TransientFETools: ∂t

θ = 0.5

u(x,t) = VectorValue(x[1],x[2])*t
u(t::Real) = x -> u(x,t)

p(x,t) = (x[1]-x[2])*t
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

U = TransientTrialFESpace(V0,u)

P = TrialFESpace(Q)

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

#
a(u,v) = inner(∇(u),∇(v))
b(v,t) = inner(v,f(t))

X = TransientMultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V0,Q])

function res(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  inner(ut,v) + q*(∇⋅u) #- inner(v,f(t)) - q*g(t)
end
function res_t(t,x,xt,y)
  u,p = x
  ut,pt = xt
  v,q = y
  0.0*inner(ut,v)
end
⋅
function jac(t,x,xt,dx,y)
  du,dp = dx
  v,q = y
  a(du,v)- (∇⋅v)*dp + q*(∇⋅du)
end

function jac_t(t,x,xt,dxt,y)
  dut,dpt = dxt
  v,q = y
  inner(dut,v)
end

function b(y)
  v,q = y
  0.0
  v⋅f(0.0) + q*g(0.0)
end

function mat(dx,y)
  du1,du2 = dx
  v1,v2 = y
  a(du1,v1)+a(du2,v2)
end

U0 = U(0.0)
P0 = P(0.0)
X0 = X(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)
ph0 = interpolate_everywhere(p(0.0),P0)
xh0 = interpolate_everywhere([uh0,ph0],X0)

# X, Y steady FESpaces
x_t = FEFunction(X0,rand(num_free_dofs(X0)))
x = FEFunction(X0,rand(num_free_dofs(X0)))
dy = get_cell_basis(Y)
dx = get_cell_basis(X0)
t1 = FETerm( (x,dy) -> res(0.0,x,x_t,dy), trian, quad )
t2 = FETerm( (x_t,dy) -> res(0.0,x,x_t,dy), trian, quad )

aaa = get_cell_values(x_t)
@show typeof(aaa)

i_to_xdual = apply(aaa) do x
  cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
  xdual = cfg.duals
  xdual
end

display(aaa[1][Block(1)])
display(aaa[1][Block(2)])
display(i_to_xdual[1][Block(1)])
display(i_to_xdual[1][Block(2)])

@show blocklength(i_to_xdual[1])
@show blocklength(aaa[1])

get_cell_jacobian(t1,x,dx,dy)
get_cell_jacobian(t2,x_t,dx,dy)

hola

t_Ω_ad = FETerm_t(res,trian,quad)
#t_Ω_ad = FETerm(res,jac,jac_t,trian,quad)
op = TransientFEOperator(X,Y,t_Ω_ad)

t0 = 0.0
tF = 1.0
dt = 0.1

ls = LUSolver()
odes = ThetaMethod(ls,dt,θ)
solver = TransientFESolver(odes)

sol_t = solve(solver,op,xh0,t0,tF)

l2(w) = w⋅w


tol = 1.0e-6
_t_n = t0

result = Base.iterate(sol_t)

for (xh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  uh_tn = xh_tn[1]
  ph_tn = xh_tn[2]
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  e = p(tn) - ph_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @test el2 < tol
end

end #module
