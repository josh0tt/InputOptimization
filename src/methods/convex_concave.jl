using JuMP
using MosekTools
using LinearAlgebra
using Gurobi
using Interpolations
using Distributions

function build_model(safe_bounds::Matrix{Float64}, Z_k::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int64, m::Int64, n_t::Int64, t_horizon::Int64, Z_cur::Matrix{Float64}, Σ::Distributions.PDMats.PDMat{Float64, Matrix{Float64}}, Q::Distributions.PDMats.PDMat{Float64, Matrix{Float64}}, scaler::UnitRangeTransform{Float64,Vector{Float64}}, delta_maxs::Vector{Float64}, method::String, sim::String)
    if method == "SDP"
        model = Model(Mosek.Optimizer)
        @variable(model, Y[1:n+m, 1:n+m], PSD)
    else
        model = Model(Gurobi.Optimizer)
    end

    @variable(model, Z_ctrl[1:n+m, 1:t_horizon])

    Z = [Z_k Z_ctrl]
    Z_diff = Z - Z_cur
    Z_diff_transpose = Z_diff'
    W_hat = Z_cur * Z_cur' + Z_cur * Z_diff_transpose + Z_diff * Z_cur'

    upper_bounds = safe_bounds[:, 2]
    lower_bounds = safe_bounds[:, 1]
    valid_bounds = .!isinf.(upper_bounds) # Vector{Bool}
    valid_indices = collect(1:n+m)[valid_bounds]

    if method == "SDP"
        @objective(model, Min, tr(Y))
        @constraint(model, [Y Diagonal(ones(n + m)); Diagonal(ones(n + m)) W_hat] >= 0, PSDCone())
        # Set MOSEK tolerances
        set_attribute(model, "INTPNT_CO_TOL_PFEAS", 1e-4)
        set_attribute(model, "INTPNT_CO_TOL_DFEAS", 1e-4)   
        set_attribute(model, "INTPNT_CO_TOL_MU_RED", 1e-4)
    elseif method == "distance"
        # penalize by how close Z is to the safe bounds
        λ = 1.0
        @objective(model, Min, -tr(W_hat) + λ * sum((Z[valid_bounds, n_t+1:(n_t+t_horizon)] .- lower_bounds[valid_bounds]).^2 + (Z[valid_bounds, n_t+1:(n_t+t_horizon)] .- upper_bounds[valid_bounds]).^2))
    else
        @objective(model, Min, -tr(W_hat))
    end

    # Constraints
    # Initial Z matching
    @constraint(model, Z[:, 1:n_t] .== Z_k[:, 1:n_t])

    # Dynamics and control constraints
    for i in n_t:(n_t+t_horizon-1)
        @constraint(model, Z[1:n, i+1] .== A_hat * Z[1:n, i] + B_hat * Z[n+1:end, i]) # Dynamics
        @constraint(model, Z[n+1:end, i] - Z[n+1:end, i+1] .<= delta_maxs) # Control variation
        @constraint(model, Z[n+1:end, i] - Z[n+1:end, i+1] .>= -delta_maxs)
    end

    # Σs = [A_hat * Σ * A_hat' + Q for _ in 1:t_horizon]
    # σs = hcat([sqrt.(diag(Σs[i])) for i in 1:t_horizon]...)
    # β = 1.96 # z-score for 95% confidence interval
    Σs = []
    push!(Σs, Σ)
    for i in 1:t_horizon-1
        Σ = A_hat * Σ * A_hat' + Q
        push!(Σs, Σ)
    end
    σs = hcat([sqrt.(diag(Σs[i])) for i in 1:t_horizon]...)
    β = 1.96 # z-score for 95% confidence interval

    # @constraint(model, lower_bounds[valid_bounds] .<= Z[valid_bounds, n_t+1:(n_t+t_horizon)])
    # @constraint(model, Z[valid_bounds, n_t+1:(n_t+t_horizon)] .<= upper_bounds[valid_bounds])
    # @constraint(model, lower_bounds[valid_bounds] .<= Z[valid_bounds, n_t+1:(n_t+t_horizon)] .- β*σs[valid_bounds])
    # @constraint(model, Z[valid_bounds, n_t+1:(n_t+t_horizon)] .+ β*σs[valid_bounds] .<= upper_bounds[valid_bounds])


    # spend equal amount of time on either side of control input
    # @constraint(model, sum(Z[n+1:end, n_t+1:end] .- Z[n+1:end, n_t]) .== zeros(m))
    # for i in n+1:n+m
    #     @constraint(model, sum(Z[i, n_t+1:(n_t+t_horizon)]) == sum(Z[i, n_t+1:(n_t+t_horizon)]))
    # end
    
    # wₖ, _ = compute_sigma(Z_k, n)
    # Σₖ = wₖ^2 * inv(Z_k * Z_k')
    # β = 1.96
    # σ = sqrt.(diag(Σₖ))
    # @constraint(model, lower_bounds[valid_bounds] .<= Z[valid_bounds, n_t+1:(n_t+t_horizon)] .- β*σ[valid_bounds])
    # @constraint(model, Z[valid_bounds, n_t+1:(n_t+t_horizon)] .+ β*σ[valid_bounds] .<= upper_bounds[valid_bounds])
    # @show valid_indices
    for i in valid_indices
        if i <= n
            # states
            @constraint(model, Z[i, n_t+1:(n_t+t_horizon)] .+ β*σs[i, :] .<= upper_bounds[i])
            @constraint(model, Z[i, n_t+1:(n_t+t_horizon)] .- β*σs[i, :] .>= lower_bounds[i])
        else
            # control inputs
            @constraint(model, Z[i, n_t+1:(n_t+t_horizon)] .<= upper_bounds[i])
            @constraint(model, Z[i, n_t+1:(n_t+t_horizon)] .>= lower_bounds[i])
        end
    end


    if sim == "xplane"
        # add constraint that we want to end up with zero roll, pitch, and yaw
        desired_end_state = zeros(n + m, 1)
        desired_end_state = StatsBase.transform(scaler, desired_end_state)
        println("Desired end state: ", desired_end_state)
        @constraint(model, Z[1:3, n_t+t_horizon] .== desired_end_state[1:3])
    end

    return model
end

function plan_control_inputs(problem::InputOptimizationProblem, method::String="approx", sim::String="aerobench")
    plan_time = time()

    safe_bounds = problem.safe_bounds
    times = problem.times
    Z_k = problem.Z
    scaler = problem.scaler
    A_hat = problem.A_hat
    B_hat = problem.B_hat
    n = problem.n
    m = problem.m
    n_t = problem.n_t
    t_horizon = problem.t_horizon
    delta_maxs = problem.delta_maxs
    
    Sigma_0_inv = diagm(0 => ones(n + m))
    if method == "SDP"
        # using random initial Z_cur leads to infeasible Mosek
        # Z_cur = [Z_k Z_k[:, end] .+ zeros(n+m, t_horizon)]
        Z_cur = [Z_k Z_k[:, end] .+ rand(problem.rng, Normal(0, 0.05), n + m, t_horizon)]
        # Z_cur = [Z_k vcat(Z_k[1:n, end] .+ zeros(n, t_horizon), markov_sequence(problem))]
    else
        Z_cur = [Z_k Z_k[:, end] .+ rand(problem.rng, Normal(0, 1.0), n + m, t_horizon)]
        # Z_cur = [Z_k vcat(Z_k[1:n, end] .+ zeros(n, t_horizon), markov_sequence(problem))]
    end    
    max_iter = 10
    tol = 1e-3
    obj = Inf
    values = Float64[]
    W_hat = Nothing
    Z = Nothing
    Z_ctrl_val = Nothing 

    build_time = time()
    model = build_model(safe_bounds, Z_k, A_hat, B_hat, n, m, n_t, t_horizon, Z_cur, problem.𝒩.Σ, problem.𝒩.Σ, scaler, delta_maxs, method, sim)
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
            if sim == "xplane"
                return nothing, true
            else
                return @error("Infeasible problem. Exiting...")
            end
        end

        # Check solver status and update Z_cur if feasible
        current_obj = objective_value(model)
        push!(values, current_obj)

        Z_ctrl_val = value.(model[:Z_ctrl])
        Z_cur = [Z_k Z_ctrl_val]
        Z = [Z_k model[:Z_ctrl]]

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
    if sim == "xplane"
        return Z_cur, false
    else
        return Z_cur
    end
end



