using JuMP
using MosekTools
using LinearAlgebra
using Gurobi
using Interpolations
using Distributions

function build_model(safe_bounds::Matrix{Float64}, Z_t::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int64, m::Int64, n_t::Int64, t_horizon::Int64, Z_cur::Matrix{Float64}, Sigma_0_inv::Matrix{Float64}, scaler::UnitRangeTransform{Float64,Vector{Float64}}, delta_maxs::Vector{Float64}, method::String)
    if method == "exact"
        model = Model(Mosek.Optimizer)
        @variable(model, Y[1:n+m, 1:n+m], PSD)
    else
        model = Model(Gurobi.Optimizer)
    end

    @variable(model, Z_ctrl[1:n+m, 1:t_horizon])

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
    @constraint(model, Z[:, 1:n_t] .== Z_t[:, 1:n_t])

    # Dynamics and control constraints
    for i in n_t:(n_t+t_horizon-1)
        @constraint(model, Z[1:n, i+1] .== A_hat * Z[1:n, i] + B_hat * Z[n+1:end, i]) # Dynamics
        @constraint(model, Z[n+1:end, i] - Z[n+1:end, i+1] .<= delta_maxs) # Control variation
        @constraint(model, Z[n+1:end, i] - Z[n+1:end, i+1] .>= -delta_maxs)
        # @constraint(model, Z[n+1:end, i+1] - Z[n+1:end, i] .<= delta_maxs)
    end

    upper_bounds = safe_bounds[:, 2]
    lower_bounds = safe_bounds[:, 1]
    valid_bounds = .!isinf.(upper_bounds)
    @constraint(model, lower_bounds[valid_bounds] .<= Z[valid_bounds, n_t+1:(n_t+t_horizon)])
    @constraint(model, Z[valid_bounds, n_t+1:(n_t+t_horizon)] .<= upper_bounds[valid_bounds])


    # add constraint that we want to end up with zero roll, pitch, and yaw
    desired_end_state = zeros(n + m, 1)
    desired_end_state = StatsBase.transform(scaler, desired_end_state)
    println("Desired end state: ", desired_end_state)
    # @constraint(model, Z[4:6, n_t+t_horizon] .== desired_end_state[4:6])
    # @constraint(model, Z[2:9, n_t+t_horizon] .== desired_end_state[2:9])

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
    n_t = problem.n_t
    t_horizon = problem.t_horizon
    delta_maxs = problem.delta_maxs
    
    Sigma_0_inv = diagm(0 => ones(n + m)) # Assuming Sigma_0_inv is defined elsewhere
    # Z_cur = [Z_t Z_t[:, end] .+ zeros(n+m, t_horizon)]
    Z_cur = [Z_t Z_t[:, end] .+ rand(Normal(0, 1.0), n + m, t_horizon)]
    max_iter = 10
    tol = 1e-3
    obj = Inf
    values = Float64[]
    W_hat = Nothing
    Z = Nothing
    Z_ctrl_val = Nothing

    build_time = time()
    model = build_model(safe_bounds, Z_t, A_hat, B_hat, n, m, n_t, t_horizon, Z_cur, Sigma_0_inv, scaler, delta_maxs, method)
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

    # unscaled_Z_ctrl = StatsBase.reconstruct(scaler, Z_ctrl_val)
    # control_traj = unscaled_Z_ctrl[n+1:end, :]'

    end_time = time()
    println("Time spent in plan_control_inputs: ", end_time - plan_time)
    # return control_traj, Z_cur, false
    return Z_cur
end



