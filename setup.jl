using JuMP, MosekTools, LinearAlgebra, StatsBase, Gurobi, Interpolations, LaTeXStrings, Distributions

function run_f16_waypoint_sim()
    pushfirst!(PyVector(pyimport("sys")."path"), "")
    # pyimport("aerobench.run_f16_sim")
    # pyimport("aerobench.visualize")
    # pyimport("aerobench.examples.waypoint.waypoint_autopilot")

    py"""
    import math
    import numpy as np
    from numpy import deg2rad

    from aerobench.run_f16_sim import run_f16_sim
    from aerobench.visualize import anim3d, plot
    from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot



    def simulate():
        'simulate the system, returning waypoints, res'

        ### Initial Conditions ###
        power = 9 # engine power level (0-10)

        # Default alpha & beta
        alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
        beta = 0                # Side slip angle (rad)

        # Initial Attitude
        alt = 3800        # altitude (ft)
        vt = 540          # initial velocity (ft/sec)
        phi = 0           # Roll angle from wings level (rad)
        theta = 0         # Pitch angle from nose level (rad)
        psi = math.pi/8   # Yaw angle from North (rad)

        # Build Initial Condition Vectors
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
        tmax = 70 # simulation time

        # make waypoint list
        e_pt = 1000
        n_pt = 3000
        h_pt = 4000

        waypoints = [[e_pt, n_pt, h_pt],
                    [e_pt + 2000, n_pt + 5000, h_pt - 100],
                    [e_pt - 2000, n_pt + 15000, h_pt - 250],
                    [e_pt - 500, n_pt + 25000, h_pt]]

        ap = WaypointAutopilot(waypoints, stdout=True)

        step = 1/30
        extended_states = True
        u_seq = np.zeros((int(tmax / step), 4))
        res = run_f16_sim(init, tmax, ap, u_seq, step=step, extended_states=extended_states, integrator_str='rk45')

        return res["times"], res["states"], np.array(res['u_list'])

    """

    # state is [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    # u is: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    times, states, controls = py"simulate"()
    
    # only keep the first 13 columns of states (the others are integration variables)
    states = states[:, 1:13]

    # only keep the first 4 columns of controls
    controls = controls[:, 1:4]

    return times, states, controls
end

function deg2rad(x)
    return π / 180 * x
end

function rad2deg(x)
    return 180 / π * x
end

"""
    scale_bounds(scaler, safe_bounds, start_index, end_index)

Scale the bounds using the given scaler.

# Arguments
- `scaler`: A `UnitRangeTransform` object representing the scaler.
- `safe_bounds`: A matrix of safe bounds.
- `start_index`: The starting index.
- `end_index`: The ending index.

# Returns
A scaled matrix of bounds.

"""
function scale_bounds(scaler::UnitRangeTransform{Float64,Vector{Float64}}, safe_bounds::Matrix{Float64}, start_index::Int, end_index::Int)
    lower_bounds = reshape(Vector{Float64}([safe_bounds[li, 1] for li in start_index:end_index]), 1, :)
    upper_bounds = reshape(Vector{Float64}([safe_bounds[ui, 2] for ui in start_index:end_index]), 1, :)

    # Set NaN values to large number
    lower_bounds[isinf.(lower_bounds)] .= -1e100
    upper_bounds[isinf.(upper_bounds)] .= 1e100

    lower_bounds = StatsBase.transform(scaler, reshape(lower_bounds, :, 1))
    upper_bounds = StatsBase.transform(scaler, reshape(upper_bounds, :, 1))

    # Replace large values with Inf
    upper_bounds[upper_bounds.>1e50] .= Inf
    lower_bounds[lower_bounds.<-1e50] .= -Inf

    return reshape(lower_bounds, :, 1), reshape(upper_bounds, :, 1)
end

"""
    estimate_linear_system(Z::Matrix{Float64}, n::Int64)

Estimates the linear system parameters A and B using the given data matrix Z and the number of inputs n.

# Arguments
- `Z::Matrix{Float64}`: The data matrix containing the input-output data.
- `n::Int64`: The number of inputs in the linear system.

# Returns
- `A::Matrix{Float64}`: The estimated system matrix A.
- `B::Matrix{Float64}`: The estimated input matrix B.
"""
function estimate_linear_system(Z::Matrix{Float64}, n::Int64)
    p = size(Z, 2) - 1
    X_prime = Z[1:n, 2:end]

    # Solve the linear system using pseudo-inverse
    Theta = X_prime * pinv(Z[:, 1:p])

    A = Theta[:, 1:n]
    B = Theta[:, n+1:end]

    return A, B
end

function build_model(safe_bounds::Matrix{Float64}, Z_t::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int64, m::Int64, t::Int64, T::Int64, Z_cur::Matrix{Float64}, Sigma_0_inv::Matrix{Float64}, scaler::UnitRangeTransform{Float64,Vector{Float64}}, delta_max::Float64, method::String)
    if method == "exact"
        model = Model(Mosek.Optimizer)
        @variable(model, Y[1:n+m, 1:n+m], PSD)
    else
        model = Model(Gurobi.Optimizer)
    end

    @variable(model, Z_ctrl[1:n+m, 1:T-t])

    Z = [Z_t Z_ctrl]
    Z_diff = Z - Z_cur
    Z_diff_transpose = Z_diff'
    W_hat = Z_cur * Z_cur' + Z_cur * Z_diff_transpose + Z_diff * Z_cur' + Sigma_0_inv

    if method == "exact"
        @objective(model, Min, tr(Y))
        @constraint(model, [Y Diagonal(ones(n + m)); Diagonal(ones(n + m)) W_hat] >= 0, PSDCone())
    else
        @objective(model, Min, -tr(W_hat))
    end

    # Constraints
    # Initial Z matching
    @constraint(model, Z[:, 1:t] .== Z_t[:, 1:t])

    # Dynamics and control constraints
    for i in t:T-1
        @constraint(model, Z[1:n, i+1] .== A_hat * Z[1:n, i] + B_hat * Z[n+1:end, i]) # Dynamics
        # @constraint(model, Z[n+1:end, i] - Z[n+1:end, i+1] .<= delta_max) # Control variation
        # @constraint(model, Z[n+1:end, i+1] - Z[n+1:end, i] .<= delta_max)
    end

    upper_bounds = safe_bounds[:, 2]
    lower_bounds = safe_bounds[:, 1]
    valid_bounds = .!isinf.(upper_bounds)
    @constraint(model, lower_bounds[valid_bounds] .<= Z[valid_bounds, t+1:T])
    @constraint(model, Z[valid_bounds, t+1:T] .<= upper_bounds[valid_bounds])


    # add constraint that we want to end up with zero roll, pitch, and yaw
    desired_end_state = zeros(n + m, 1)
    desired_end_state = StatsBase.transform(scaler, desired_end_state)
    println("Desired end state: ", desired_end_state)
    @constraint(model, Z[4:6, T] .== desired_end_state[4:6])

    return model
end

function plan_control_inputs(problem::InputOptimizationProblem, method::String="approx")
    plan_time = time()

    safe_bounds = problem.safe_bounds
    times = problem.times
    Z_t = problem.Z
    scaler = problem.scaler
    A_hat = problem.A_hat
    B_hat = problem.B_hat
    n = problem.n
    m = problem.m
    t = problem.t
    t_horizon = problem.t_horizon
    
    T = t + t_horizon

    Sigma_0_inv = diagm(0 => ones(n + m)) # Assuming Sigma_0_inv is defined elsewhere
    # Z_cur = [Z_t Z_t[:, end] .+ zeros(n+m, t_horizon)]
    Z_cur = [Z_t Z_t[:, end] .+ rand(Normal(0, 1.0), n + m, t_horizon)]
    max_iter = 10
    delta_max = 1.0
    tol = 1e-3
    obj = Inf
    values = Float64[]
    W_hat = Nothing
    Z = Nothing
    Z_ctrl_val = Nothing

    build_time = time()
    model = build_model(safe_bounds, Z_t, A_hat, B_hat, n, m, t, T, Z_cur, Sigma_0_inv, scaler, delta_max, method)
    build_end_time = time()
    println("Time spent in build_model: ", build_end_time - build_time)

    for iter in 1:max_iter
        println("#"^30)
        println("Iteration $iter/$max_iter")
        println("#"^30)

        # Solve the problem
        JuMP.optimize!(model)

        # Check solver status and update Z_cur if feasible
        println("Solver status: ", termination_status(model))
        if termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED
            println("Infeasible problem. Stabilizing aircraft...")
            @warn("Infeasible problem. Stabilizing aircraft...")
            # stable_control_traj = stabilize_aircraft(safe_bounds, times, Z_t, scaler, start_time, t0, t_horizon, n, m, t, A_hat, B_hat, delta_max)
            # Z_ctrl_val = zeros(n + m, t_horizon)
            # Z_ctrl_val[n+1:end, :] .= stable_control_traj
            # unscaled_Z_ctrl = StatsBase.reconstruct(scaler, Z_ctrl_val)
            # control_traj = unscaled_Z_ctrl[n+1:end, :]'

            # return control_traj, zeros(n + m, T), execution_times, true
        end

        # Check solver status and update Z_cur if feasible
        current_obj = objective_value(model)
        push!(values, current_obj)

        Z_ctrl_val = value.(model[:Z_ctrl])
        Z_cur = [Z_t Z_ctrl_val]
        Z = [Z_t model[:Z_ctrl]]

        # Update W_hat for model objective 
        Z_diff = Z - Z_cur
        Z_diff_transpose = Z_diff'
        W_hat = Z_cur * Z_cur' + Z_cur * Z_diff_transpose + Z_diff * Z_cur' + Sigma_0_inv
        set_objective_function(model, -tr(W_hat))


        # Convergence check
        if abs(obj - current_obj) < tol
            println("Converged after $iter iterations")
            break
        end
        obj = current_obj

    end

    unscaled_Z_ctrl = StatsBase.reconstruct(scaler, Z_ctrl_val)
    control_traj = unscaled_Z_ctrl[n+1:end, :]'

    end_time = time()
    println("Time spent in plan_control_inputs: ", end_time - plan_time)
    return control_traj, Z_cur, false
end

# plts = []
# labels = ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow"]
# for i in 1:length(labels)
#     if i in collect(2:9)
#         # convert rad to deg
#         plt = plot(times, rad2deg(Z_unscaled[i, :]), label=labels[i])
#     else
#         plt = plot(times, Z_unscaled[i, :], label=labels[i])
#     end
#     push!(plts, plt)
# end
# plt = plot(plts..., layout=(7, 2), size=(800, 800), margin=5mm)
# savefig("/Users/joshuaott/Downloads/plot.pdf")

# # plot controls 
# plts = []
# labels = ["throt", "ele", "ail", "rud"]
# for i in 17:20
#     plt = plot(times, Z_unscaled[i, :], label=labels[i-16])
#     push!(plts, plt)
# end
# plt = plot(plts..., layout=(4, 1), size=(800, 800), margin=5mm)
# savefig("/Users/joshuaott/Downloads/plot_controls.pdf")
