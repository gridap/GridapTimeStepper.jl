module HeatEquationWithAutoDiffTests

using Gridap
using ForwardDiff
using LinearAlgebra
using Test
using GridapODEs.ODETools
using GridapODEs.TransientFETools
using GridapODEs.TransientFETools
using Gridap.FESpaces: get_algebraic_operator
using Gridap.FESpaces
using Gridap.Arrays

import Gridap: ∇
import GridapODEs.TransientFETools: ∂t

θ = 0.2

u(x,t) = (1.0-x[1])*x[1]*(1.0-x[2])*x[2]*t
u(t::Real) = x -> u(x,t)
f(t) = x -> ∂t(u)(x,t)-Δ(u(t))(x)

domain = (0,1,0,1)
partition = (2,2)
model = CartesianDiscreteModel(domain,partition)

order = 2

V0 = FESpace(
  reffe=:Lagrangian, order=order, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")
U = TransientTrialFESpace(V0,u)

trian = Triangulation(model)
degree = 2*order
quad = CellQuadrature(trian,degree)

#
a(u,v) = ∇(v)⋅∇(u)
b(v,t) = v*f(t)

res(t,u,ut,v) = a(u,v) + ut*v - b(v,t)
jac(t,u,ut,du,v) = a(du,v)
jac_t(t,u,ut,dut,v) = dut*v

t_Ω = FETerm(res,jac,jac_t,trian,quad)
t_Ω_ad = FETerm_t(res,trian,quad)
op = TransientFEOperator(U,V0,t_Ω_ad)

Uh = evaluate(U,nothing)
du_t = get_cell_basis(Uh)
dv = get_cell_basis(V0)
du = get_cell_basis(Uh)
uh = FEFunction(Uh,rand(num_free_dofs(Uh)))

cell_r = get_cell_residual(t_Ω,0.5,uh,uh,dv)
cell_j = get_cell_jacobian(t_Ω,0.5,uh,uh,du,dv)
cell_j_t = TransientFETools.get_cell_jacobian_t(t_Ω,0.5,uh,uh,du,dv,0.1)

cell_r_ad = get_cell_residual(t_Ω_ad,0.5,uh,uh,dv)
cell_j_ad = get_cell_jacobian(t_Ω_ad,0.5,uh,uh,du,dv)
cell_j_t_ad = TransientFETools.get_cell_jacobian_t(t_Ω_ad,0.5,uh,uh,du,dv,0.1)

test_array(cell_r_ad,cell_r)
test_array(cell_j_ad,cell_j)
test_array(cell_j_t_ad,cell_j_t)

t0 = 0.0
tF = 1.0
dt = 0.1

U0 = U(0.0)
uh0 = interpolate_everywhere(u(0.0),U0)

ls = LUSolver()
using Gridap.Algebra: NewtonRaphsonSolver
nls = NLSolver(ls;show_trace=true,method=:newton) #linesearch=BackTracking())
odes = ThetaMethod(ls,dt,θ)
solver = TransientFESolver(odes)

sol_t = solve(solver,op,uh0,t0,tF)

# Juno.@enter Base.iterate(sol_t)

l2(w) = w*w

tol = 1.0e-6
_t_n = t0

for (uh_tn, tn) in sol_t
  global _t_n
  _t_n += dt
  e = u(tn) - uh_tn
  el2 = sqrt(sum( integrate(l2(e),trian,quad) ))
  @test el2 < tol
end

end #module
