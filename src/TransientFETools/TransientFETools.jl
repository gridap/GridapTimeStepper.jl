module TransientFETools

using Test

using Gridap.Helpers

export ∂t

import GridapODEs.ODETools: ∂t
import GridapODEs.ODETools: time_derivative

export TransientTrialFESpace
export TransientMultiFieldFESpace
export test_transient_trial_fe_space
import Gridap.Fields: evaluate
import Gridap.Fields: evaluate!
import Gridap.MultiField: MultiFieldFESpace
using Gridap.FESpaces: FESpace
using Gridap.FESpaces: SingleFieldFESpace
using Gridap.FESpaces: TrialFESpace
using Gridap.FESpaces: ZeroMeanFESpace
using Gridap.FESpaces: get_free_values
using Gridap.FESpaces: get_dirichlet_values
using Gridap.FESpaces: TrialFESpace!
using Gridap.FESpaces: HomogeneousTrialFESpace

export TransientFETerm
export TransientAffineFETerm
export TransientConstantFETerm
export FETerm_t
import Gridap.FESpaces: FETerm
import Gridap.FESpaces: get_cell_residual
import Gridap.FESpaces: get_cell_jacobian
import Gridap.FESpaces: get_cell_values
import Gridap.Geometry: Triangulation
import Gridap.Geometry: CellQuadrature
using Gridap.FESpaces: restrict
using Gridap.FESpaces: integrate
import Gridap.Geometry: get_cell_id
using Gridap.FESpaces: _push_matrix_contribution!
using Gridap.FESpaces: _push_vector_contribution!
using Gridap.FESpaces: autodiff_cell_jacobian_from_residual

export TransientFEOperator
export TransientAffineFEOperator
export TransientConstantFEOperator
using Gridap.FESpaces: Assembler
using Gridap.FESpaces: SparseMatrixAssembler
import GridapODEs.ODETools: allocate_cache
import GridapODEs.ODETools: update_cache!
import GridapODEs.ODETools: ODEOperator
import GridapODEs.ODETools: AffineODEOperator
import GridapODEs.ODETools: ConstantODEOperator
import GridapODEs.ODETools: allocate_residual
import GridapODEs.ODETools: allocate_jacobian
import GridapODEs.ODETools: residual!
import GridapODEs.ODETools: jacobian!
import GridapODEs.ODETools: jacobian_t!
import GridapODEs.ODETools: jacobian_and_jacobian_t!
import GridapODEs.ODETools: OperatorType
using GridapODEs.ODETools: Nonlinear
using GridapODEs.ODETools: Affine
using GridapODEs.ODETools: Constant
import Gridap.FESpaces: get_algebraic_operator
import Gridap.FESpaces: collect_cell_residual
import Gridap.FESpaces: collect_cell_jacobian
import Gridap.FESpaces: assemble_vector!
import Gridap.FESpaces: assemble_matrix_add!
import Gridap.FESpaces: allocate_vector
import Gridap.FESpaces: allocate_matrix
using Gridap.FESpaces: is_a_fe_function
using Gridap.FESpaces: is_a_fe_cell_basis
using Gridap.FESpaces: get_cell_basis
using GridapODEs.ODETools: test_ode_operator
export test_transient_fe_operator

export TransientFESolver
import Gridap.FESpaces: FESolver
import GridapODEs.ODETools: ODESolver
import Gridap: solve
import Gridap.Algebra: solve!
import GridapODEs.ODETools: solve_step!
export test_transient_fe_solver

export TransientFEFunction
import Gridap.FESpaces: FEFunction
import Gridap.FESpaces: EvaluationFunction

export TransientFESolution
import Gridap: solve
import GridapODEs.ODETools: ODESolution
import GridapODEs.ODETools: GenericODESolution
import Base: iterate
export test_transient_fe_solution

include("TransientFESpaces.jl")

include("TransientFETerms.jl")

include("TransientFEOperators.jl")

include("ODEOperatorInterfaces.jl")

include("TransientFESolvers.jl")

include("TransientFESolutions.jl")

include("TransientFETermsWithAutoDiff.jl")

end #module
