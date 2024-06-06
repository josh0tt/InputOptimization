module InputOptimization

using Parameters
using PyCall
using StatsBase
using Plots
using Measures
using Random
using Distributions

abstract type SolutionMethod end
struct ConvexConcave <: SolutionMethod end
struct ConvexConcaveSDP <: SolutionMethod end
struct OrthogonalMultisine <: SolutionMethod end
struct RandomSequence <: SolutionMethod end

@with_kw mutable struct InputOptimizationProblem
    rng::MersenneTwister                                # random number generator
    Z::Matrix{Float64}                                  # (n+m) x t observed data
    scaler::UnitRangeTransform{Float64,Vector{Float64}} # scaler used to scale the data
    times::Vector{Float64}                              # t x 1 times at which data is observed
    A_hat::Matrix{Float64}                              # n x n estimated state matrix
    B_hat::Matrix{Float64}                              # n x m estimated control matrix
    𝒩::FullNormal                                       # MVnormal distribution for process noise xp - Âx - B̂u
    n::Int                                              # number of state variables
    m::Int                                              # number of control variables
    n_t::Int                                            # number of time steps
    t_horizon::Int                                      # planning horizon in number of time steps
    Δt::Float64                                         # time step for planning
    safe_bounds::Matrix{Float64}                        # (n+m) x 2 safety bounds scaled
    safe_bounds_unscaled::Matrix{Float64}               # (n+m) x 2 safety bounds unscaled
    delta_maxs::Vector{Float64}                         # maximum change in control inputs between time steps
    max_As::Vector{Float64}                             # maximum amplitude of control inputs unscaled
    f_min::Float64                                      # minimum frequency for multisines
    f_max::Float64                                      # maximum frequency for multisines
    row_names::Vector{String}                           # Z row names
    equal_time_constraint::Bool                         # equal amount of time on either side of control input for ConvexConcave
end

include("DMDc.jl")
include("run_cylinder.jl")
include("run_f16.jl")
include("setup.jl")
include("utilities/helpers.jl")
include("methods/convex_concave.jl")
include("methods/orthogonal_multisines.jl")
include("methods/random.jl")
include("utilities/plotting.jl")
include("utilities/xplane_utils.jl")
include("run_experiments.jl")
include("run_xplane.jl")


function solve(problem::InputOptimizationProblem, method::ConvexConcave)::Matrix{Float64}
    println("Solving with Convex Concave")
    Z_planned = plan_control_inputs(problem)
    return Z_planned
end

function solve(problem::InputOptimizationProblem, method::ConvexConcaveSDP)::Matrix{Float64}
    println("Solving with Convex Concave SDP")
    Z_planned = plan_control_inputs(problem, "SDP")
    return Z_planned
end

function solve(problem::InputOptimizationProblem, method::OrthogonalMultisine)::Matrix{Float64}
    println("Solving with Orthogonal Multisine")
    Z_planned = run_orthogonal_multisines(problem)
    return Z_planned
end

function solve(problem::InputOptimizationProblem, method::RandomSequence)::Matrix{Float64}
    println("Solving with Random Sequence")
    Z_planned = run_random(problem)
    return Z_planned
end

export InputOptimizationProblem
       ConvexConcave, 
       ConvexConcaveSDP, 
       OrthogonalMultisine, 
       RandomSequence, 
       solve,
       f16_problem_setup,
       cylinder_problem_setup,
       CylinderFlowData,
       DMDc,
       project_down,
       project_up,
       find_max_As, 
       scale_bounds,
       estimate_linear_system,
       fit_process_noise,
       run_f16_sim,
       run_f16_waypoint_sim,
       make_gifs,
       SimData,
       run_f16_experiments,
       run_xplane,
       run_cylinder,
       run_cylinder_planned_inputs

end