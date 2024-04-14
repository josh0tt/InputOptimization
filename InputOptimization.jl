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

function problem_setup()
    safe_bounds = [
        300 2500; # vt ft/s
        deg2rad(-10) deg2rad(45); # alpha
        deg2rad(-30) deg2rad(30); # beta 
        deg2rad(-90) deg2rad(90); # phi
        deg2rad(-30) deg2rad(30); # theta
        deg2rad(-180) deg2rad(180); # psi
        deg2rad(-180) deg2rad(180); # P
        deg2rad(-180) deg2rad(180); # Q
        deg2rad(-180) deg2rad(180); # R
        -Inf Inf; # pn ft
        -Inf Inf; # pe ft
        1000 50000; # h ft
        0 100; # pow
        0 1; # throt
        deg2rad(-10) deg2rad(10); # ele
        deg2rad(-15) deg2rad(15); # ail
        deg2rad(-10) deg2rad(10) # rud
    ] 


    # 1. run F16 waypoint simulation to collect data set 
    times, states, controls = run_f16_waypoint_sim()
    n, m = size(states, 2), size(controls, 2)
    t = length(times)
    Δt = times[2] - times[1]
    @show Δt
    t_horizon = round(Int64, 25 / Δt)
    @show t_horizon


    # 2. scale the data
    Z_unscaled = hcat(states, controls)' # Z is shaped as (n+m,t) where n is the number of states and m is the number of controls
    scaler = fit(UnitRangeTransform, Z_unscaled, dims=2)
    Z = StatsBase.transform(scaler, Z_unscaled)

    # scale the bounds as well
    lower_bounds, upper_bounds = scale_bounds(scaler, safe_bounds, 1, n + m)

    new_safe_bounds = zeros(size(safe_bounds))
    for i in 1:size(safe_bounds, 1)
        new_safe_bounds[i, 1] = lower_bounds[i]
        new_safe_bounds[i, 2] = upper_bounds[i]
    end
    safe_bounds = new_safe_bounds

    # 3. estimate the linear system
    A_hat, B_hat = estimate_linear_system(Z, n)

    # 4. create the InputOptimizationProblem
    problem = InputOptimizationProblem(Z, scaler, times, A_hat, B_hat, n, m, t, t_horizon, Δt, safe_bounds, ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow", "throt", "ele", "ail", "rud"])

    return problem
end

function solve(problem::InputOptimizationProblem, method::ConvexConcave)
    println("Solving ConvexConcaveProblem")

    control_traj, Z_planned, infeasible_flag = plan_control_inputs(problem)

    return control_traj, Z_planned, infeasible_flag
end

problem = problem_setup()
solve(problem, ConvexConcave())
