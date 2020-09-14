struct TransientFETermFromIntegrationWithAutoDiff <: TransientFETerm
  res::Function
  trian::Triangulation
  quad::CellQuadrature
end

function FETerm_t(
  res::Function, trian::Triangulation, quad::CellQuadrature)
  TransientFETermFromIntegrationWithAutoDiff(res,trian,quad)
end

function get_cell_residual(tr::TransientFETermFromIntegrationWithAutoDiff,t::Real,uh,uh_t,v)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  _v = restrict(v,tr.trian)
  _uh = restrict(uh,tr.trian)
  _uh_t = restrict(uh_t,tr.trian)
  integrate(tr.res(t,_uh,_uh_t,_v),tr.trian,tr.quad)
end

function get_cell_jacobian(tr::TransientFETermFromIntegrationWithAutoDiff,t::Real,uh,uh_t,du,v)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  @assert is_a_fe_cell_basis(du)
  _v = restrict(v,tr.trian)
  _uh_t = restrict(uh_t,tr.trian)  
  function uh_to_cell_residual(uh)
    _uh = restrict(uh,tr.trian)
    integrate(tr.res(t,_uh,_uh_t,_v),tr.trian,tr.quad)
  end
  cell_j = autodiff_cell_jacobian_from_residual(uh_to_cell_residual,uh,get_cell_id(tr))
  cell_j
end

function get_cell_jacobian_t(tr::TransientFETermFromIntegrationWithAutoDiff,t::Real,uh,uh_t,du_t,v,duht_du::Real)
  @assert is_a_fe_function(uh)
  @assert is_a_fe_function(uh_t)
  @assert is_a_fe_cell_basis(v)
  @assert is_a_fe_cell_basis(du_t)
  _v = restrict(v,tr.trian)
  _uh = restrict(uh,tr.trian)
  _du_t = restrict(du_t,tr.trian)  
  function uh_to_cell_residual(uh_t)
    _uh_t = restrict(uh_t,tr.trian)
    integrate(duht_du*tr.res(t,_uh,_uh_t,_v),tr.trian,tr.quad)
  end
  cell_j_t = autodiff_cell_jacobian_from_residual(uh_to_cell_residual,uh_t,get_cell_id(tr))
  cell_j_t
  #integrate(duht_du*tr.jac_t(t,_uh,_uh_t,_du_t,_v),tr.trian,tr.quad)
end

function get_cell_id(t::TransientFETermFromIntegrationWithAutoDiff)
  get_cell_id(t.trian)
end