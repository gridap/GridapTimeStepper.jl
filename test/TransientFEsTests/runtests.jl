module TransientFEToolsTests

using Test

@testset "TransientFETests" begin include("TransientFETests.jl") end

@testset "TransientFEOperatorsTests" begin include("TransientFEOperatorsTests.jl") end

@testset "AffineFEOperatorsTests" begin include("AffineFEOperatorsTests.jl") end

@testset "ConstantFEOperatorsTests" begin include("ConstantFEOperatorsTests.jl") end

@testset "HeatEquationTests" begin include("HeatEquationTests.jl") end

@testset "HeatEquationWithAutoDiffTests" begin include("HeatEquationWithAutoDiffTests.jl") end

@testset "HeatVectorEquationTests" begin include("HeatVectorEquationTests.jl") end

@testset "VectorHeatEquationTests" begin include("VectorHeatEquationTests.jl") end

@testset "StokesEquationTests" begin include("StokesEquationTests.jl") end

@testset "StokesEquationWithAutoDiffTests" begin include("StokesEquationWithAutoDiffTests.jl") end

@testset "BoundaryEquationTests" begin include("BoundaryHeatEquationTests.jl") end

@testset "DGHeatEquationTests" begin include("DGHeatEquationTests.jl") end

end # module
