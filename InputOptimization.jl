using Parameters
using PyCall
using StatsBase
using Plots
using Measures

abstract type SolutionMethod end
struct ConvexConcave <: SolutionMethod end
struct OrthogonalMultisine <: SolutionMethod end

@with_kw struct InputOptimizationProblem
    Z::Matrix{Float64}                                  # t x (n+m) observed data
    scaler::UnitRangeTransform{Float64,Vector{Float64}} # scaler used to scale the data
    times::Vector{Float64}                              # t x 1 times at which data is observed
    A_hat::Matrix{Float64}                              # n x n estimated state matrix
    B_hat::Matrix{Float64}                              # n x m estimated control matrix
    n::Int                                              # number of state variables
    m::Int                                              # number of control variables
    t::Int                                              # number of time steps
    t_horizon::Int                                      # planning horizon
    Δt::Float64                                         # time step for planning
    safe_bounds::Matrix{Float64}                        # (n+m) x 2 safety bounds
    column_names::Vector{String}                        # Z column names
end

include("setup.jl")
include("convex_concave.jl")
include("plotting.jl")

function solve(problem::InputOptimizationProblem, method::ConvexConcave)
    println("Solving with Convex Concave")

    control_traj, Z_planned, infeasible_flag = plan_control_inputs(problem)

    return control_traj, Z_planned, infeasible_flag
end

function solve(problem::InputOptimizationProblem, method::OrthogonalMultisine)
    println("Solving with Orthogonal Multisine")

    times, states, controls = run_orthogonal_multisines(problem.times, problem.t_horizon, problem.t, problem.Z[1:problem.t, problem.n+1:end], problem.t_horizon, problem.m)

    return times, states, controls
end

problem = problem_setup()
control_traj, Z_planned, infeasible_flag = solve(problem, ConvexConcave())

plot(problem, Z_planned)
