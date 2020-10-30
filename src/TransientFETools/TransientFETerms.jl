function get_cell_residual(fet::FETerm,t,uh,uh_t,v)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  get_cell_residual(fet,uh,v)
end

function get_cell_jacobian(fet::FETerm,t,uh,uh_t,du,v)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  @assert is_a_fe_cell_basis(du)
  get_cell_jacobian(fet,uh,du,v)
end

function get_cell_jacobian_t(fet::FETerm,t,uh,uh_t,du_t,v,duht_du)
  nothing
end

abstract type TransientFETerm end

get_cell_residual(::TransientFETerm,t,uh,uh_t,v) = @notimplemented
get_cell_jacobian(::TransientFETerm,t,uh,uh_t,du_t,v) = @notimplemented
get_cell_jacobian_t(::TransientFETerm,t,uh,uh_t,du_t,v,duht_du) = @notimplemented
get_cell_values(tr::TransientFETerm,uhd) = @notimplemented
get_cell_id(tr::TransientFETerm) = @notimplemented

struct TransientFETermFromIntegration <: TransientFETerm
  res::Function
  jac::Function
  jac_t::Function
  trian::Triangulation
  quad::CellQuadrature
end

function TransientFETerm(
  res::Function, jac::Function, jac_t::Function, trian::Triangulation, quad::CellQuadrature)
  TransientFETermFromIntegration(res,jac,jac_t,trian,quad)
end

function FETerm(
  res::Function, jac::Function, jac_t::Function, trian::Triangulation, quad::CellQuadrature)
  TransientFETerm(res,jac,jac_t,trian,quad)
end

function get_cell_residual(tr::TransientFETermFromIntegration,t::Real,uh,uh_t,v)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  _v = restrict(v,tr.trian)
  _uh = restrict(uh,tr.trian)
  _uh_t = restrict(uh_t,tr.trian)
  integrate(tr.res(t,_uh,_uh_t,_v),tr.trian,tr.quad)
end

function get_cell_jacobian(tr::TransientFETermFromIntegration,t::Real,uh,uh_t,du,v)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  @assert is_a_fe_cell_basis(du)
  _v = restrict(v,tr.trian)
  _uh = restrict(uh,tr.trian)
  _uh_t = restrict(uh_t,tr.trian)
  _du = restrict(du,tr.trian)
  integrate(tr.jac(t,_uh,_uh_t,_du,_v),tr.trian,tr.quad)
end

function get_cell_jacobian_t(tr::TransientFETermFromIntegration,t::Real,uh,uh_t,du_t,v,duht_du::Real)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  @assert is_a_fe_cell_basis(du_t)
  _v = restrict(v,tr.trian)
  _uh = restrict(uh,tr.trian)
  _uh_t = restrict(uh_t,tr.trian)
  _du_t = restrict(du_t,tr.trian)
  integrate(duht_du*tr.jac_t(t,_uh,_uh_t,_du_t,_v),tr.trian,tr.quad)
end

function get_cell_values(tr::TransientFETermFromIntegration,uhd)
  @assert is_a_fe_function(uhd)
  cellvals = get_cell_values(uhd)
  reindex(cellvals,tr.trian)
end

function get_cell_id(t::TransientFETermFromIntegration)
  get_cell_id(t.trian)
end

function collect_cell_residual(t::Real,uh,uh_t,v,terms)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  w = []
  r = []
  for term in terms
    cellvals = get_cell_residual(term,t,uh,uh_t,v)
    cellids = get_cell_id(term)
    _push_vector_contribution!(w,r,cellvals,cellids)
  end
  (w,r)
end

function collect_cell_jacobian(t::Real,uh,uh_t,du,v,terms)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  @assert is_a_fe_cell_basis(du)
  w = []
  r = []
  c = []
  for term in terms
    cellvals = get_cell_jacobian(term,t,uh,uh_t,du,v)
    cellids = get_cell_id(term)
    _push_matrix_contribution!(w,r,c,cellvals,cellids)
  end
  (w,r,c)
end

function collect_cell_jacobian_t(t::Real,uh,uh_t,du,v,duht_du::Real,terms)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  @assert is_a_fe_cell_basis(du)
  w = []
  r = []
  c = []
  for term in terms
    cellvals = get_cell_jacobian_t(term,t,uh,uh_t,du,v,duht_du)
    cellids = get_cell_id(term)
    _push_matrix_contribution!(w,r,c,cellvals,cellids)
  end
  (w,r,c)
end

# We need _push_vector and _push_matrix contribution
# internal methods from Gridap

# function _push_matrix_contribution!(w,r,c,cellvals,cellids)
#   push!(w,cellvals)
#   push!(r,cellids)
#   push!(c,cellids)
#   nothing
# end
#
# function _push_matrix_contribution!(w,r,c,cellvals::Nothing,cellids)
#   nothing
# end
#
# function _push_matrix_contribution!(w,r,c,cellvals::SkeletonCellMatrix,cellids::SkeletonPair)
#   push!(w,cellvals.ll)
#   push!(w,cellvals.lr)
#   push!(w,cellvals.rl)
#   push!(w,cellvals.rr)
#   push!(r,cellids.left); push!(c,cellids.left)
#   push!(r,cellids.left); push!(c,cellids.right)
#   push!(r,cellids.right); push!(c,cellids.left)
#   push!(r,cellids.right); push!(c,cellids.right)
#   nothing
# end
#
# function _push_vector_contribution!(v,r,cellvals,cellids)
#   push!(v,cellvals)
#   push!(r,cellids)
#   nothing
# end
#
# function _push_vector_contribution!(v,r,cellvals::Nothing,cellids)
#   nothing
# end
#
# function _push_vector_contribution!(v,r,cellvals::SkeletonCellVector,cellids::SkeletonPair)
#   push!(v,cellvals.left)
#   push!(v,cellvals.right)
#   push!(r,cellids.left)
#   push!(r,cellids.right)
#   nothing
# end

"""
Alternative constructor for affine operators
"""
function TransientAffineFETerm(
  m::Function,a::Function,b::Function,trian::Triangulation,quad::CellQuadrature)
  res(t,u,ut,v) = m(t,ut,v) + a(t,u,v) - b(t,v)
  jac(t,u,ut,du,v) = a(t,du,v)
  jac_t(t,u,ut,dut,v) = m(t,dut,v)
  FETerm(res,jac,jac_t,trian,quad)
end

"""
Alternative constructor for constant operators
"""
function TransientConstantFETerm(
  m::Function,a::Function,b::Function,trian::Triangulation,quad::CellQuadrature)
  res(t,u,ut,v) = m(ut,v) + a(u,v) - b(v)
  jac(t,u,ut,du,v) = a(du,v)
  jac_t(t,u,ut,dut,v) = m(dut,v)
  FETerm(res,jac,jac_t,trian,quad)
end
