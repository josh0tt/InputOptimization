using JuMP
using MosekTools
using LinearAlgebra
using Gurobi
using Interpolations
using Distributions

function build_model(safe_bounds::Matrix{Float64}, Z_k::Matrix{Float64}, X′::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, 
                     n::Int64, m::Int64, n_t::Int64, t_horizon::Int64, Z_cur::Matrix{Float64}, 
                     Σ::Distributions.PDMats.PDMat{Float64, Matrix{Float64}}, Q::Distributions.PDMats.PDMat{Float64, Matrix{Float64}}, 
                     scaler::UnitRangeTransform{Float64,Vector{Float64}}, delta_maxs::Vector{Float64}, 
                     method::String, sim::String, equal_time_constraint::Bool)
    
    if method == "SDP"
        model = Model(Mosek.Optimizer)
        @variable(model, Y[1:n+m, 1:n+m], Symmetric)
    else
        model = Model(Gurobi.Optimizer)
    end

    @variable(model, Z_ctrl[1:n+m, 1:t_horizon])

    Z = [Z_k Z_ctrl]
    Z_diff = Z - Z_cur
    Z_diff_transpose = Z_diff'
    Z_cur_Z_diff_transpose = Z_cur * Z_diff_transpose
    W_hat = Z_cur * Z_cur' + Z_cur_Z_diff_transpose + Z_cur_Z_diff_transpose'

    upper_bounds = safe_bounds[:, 2]
    lower_bounds = safe_bounds[:, 1]
    valid_bounds = .!isinf.(upper_bounds) # Vector{Bool}
    valid_indices = collect(1:n+m)[valid_bounds]

    if method == "SDP"
        @objective(model, Min, tr(Y))
        @constraint(model, [Y Diagonal(ones(n + m)); Diagonal(ones(n + m)) W_hat] >= 0, PSDCone())
    else
        @objective(model, Min, -tr(W_hat))
    end

    # Constraints
    # Dynamics and control constraints
    # first state of Z_ctrl must be equal to the last collected state at k+1
    @constraint(model, Z_ctrl[1:n, 1] .== X′[:, end]) 
    @constraint(model, Z_k[n+1:end, end] - Z_ctrl[n+1:end, 1] .<= delta_maxs) # Control variation on uₖ₊₁
    @constraint(model, Z_k[n+1:end, end] - Z_ctrl[n+1:end, 1] .>= -delta_maxs)
    for i in 1:(t_horizon-1)
        @constraint(model, Z_ctrl[1:n, i+1] .== A_hat * Z_ctrl[1:n, i] + B_hat * Z_ctrl[n+1:end, i]) # Dynamics
        @constraint(model, Z_ctrl[n+1:end, i] - Z_ctrl[n+1:end, i+1] .<= delta_maxs) # Control variation
        @constraint(model, Z_ctrl[n+1:end, i] - Z_ctrl[n+1:end, i+1] .>= -delta_maxs)
    end
    # account for last time step of control 
    @constraint(model, Z_ctrl[n+1:end, t_horizon] .== Z_ctrl[n+1:end, t_horizon-1]) 

    # Compute Σs and σs
    Σs = Vector{Matrix{Float64}}(undef, t_horizon)
    Σs[1] = Σ
    for i in 2:t_horizon
        Σs[i] = A_hat * Σs[i-1] * A_hat' + Q
    end
    σs = hcat([sqrt.(diag(Σs[i])) for i in 1:t_horizon]...)
    β = 1.96#2.576  # z-score for 99% confidence interval

    # spend equal amount of time on either side of control input
    if equal_time_constraint
        @constraint(model, sum(Z_ctrl[n+1:end, 1:end] .- Z_ctrl[n+1:end, 1]) .== zeros(m))
    end
    
    @show valid_indices
    for i in valid_indices
        if i <= n
            # states
            @constraint(model, Z_ctrl[i, 1:t_horizon] .+ β*σs[i, :] .<= upper_bounds[i])
            @constraint(model, Z_ctrl[i, 1:t_horizon] .- β*σs[i, :] .>= lower_bounds[i])
        else
            # control inputs
            @constraint(model, Z_ctrl[i, 1:t_horizon] .<= upper_bounds[i])
            @constraint(model, Z_ctrl[i, 1:t_horizon] .>= lower_bounds[i])
        end
    end


    if sim == "xplane"
        # add constraint that we want to end up with zero roll, pitch, and yaw
        desired_end_state = StatsBase.transform(scaler, zeros(n + m, 1))
        println("Desired end state: ", desired_end_state)
        # @constraint(model, Z[1:3, n_t+t_horizon] .== desired_end_state[1:3])
        @constraint(model, Z_ctrl[1:3, t_horizon] .== desired_end_state[1:3])
    end

    return model
end

function plan_control_inputs(problem::InputOptimizationProblem, 
                             method::String="approx", 
                             sim::String="aerobench")

    plan_time = time()
    safe_bounds = problem.safe_bounds
    times = problem.times
    Z_k = problem.Z
    X′ = problem.X′
    scaler = problem.scaler
    A_hat = problem.A_hat
    B_hat = problem.B_hat
    n = problem.n
    m = problem.m
    n_t = problem.n_t
    t_horizon = problem.t_horizon
    delta_maxs = problem.delta_maxs
    
    # Feasible point initialization
    Z_init = [Z_k zeros(n+m, t_horizon)]
    for i in problem.n+1:problem.n+m
        A = rand(problem.rng, Normal(0.2, 0.1))*(maximum(Z_init[i, 1:n_t]) - minimum(Z_init[i, 1:n_t]))
        w = rand(problem.rng, Normal(0.1, 0.05))
        Z_init[i, n_t+1:end] .= A*sin.(w .* range(1, t_horizon, step=1)) .+ Z_init[i, n_t]
    end
    for t in 1:t_horizon
        Z_init[1:n, n_t+t] .= problem.A_hat * Z_init[1:n, n_t+t-1] + problem.B_hat * Z_init[n+1:end, n_t+t-1]
    end
    Z_cur = Z_init
    
    max_iter = method == "SDP" ? 10 : 3
    tol = 1e-3
    obj = Inf
    values = Float64[]
    W_hat = Nothing
    Z = Nothing
    Z_ctrl_val = Nothing 

    build_time = time()
    model = build_model(safe_bounds, Z_k, X′, A_hat, B_hat, n, m, n_t, t_horizon, Z_cur, problem.𝒩.Σ, problem.𝒩.Σ, scaler, delta_maxs, method, sim, problem.equal_time_constraint)
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
        # Z_diff = Z - Z_cur
        # Z_diff_transpose = Z_diff'
        # Z_cur_Z_diff_transpose = Z_cur * Z_diff_transpose
        # W_hat = Z_cur * Z_cur' + Z_cur_Z_diff_transpose + Z_cur_Z_diff_transpose'
        # # W_hat = Z_cur * Z_cur' + Z_cur * Z_diff_transpose + Z_diff * Z_cur'

        # if method == "SDP"
        #     @objective(model, Min, tr(Y))
        #     @constraint(model, [Y Diagonal(ones(n + m)); Diagonal(ones(n + m)) W_hat] >= 0, PSDCone())
        #     # Set MOSEK tolerances
        #     # set_attribute(model, "INTPNT_CO_TOL_PFEAS", 1e-4)
        #     # set_attribute(model, "INTPNT_CO_TOL_DFEAS", 1e-4)   
        #     # set_attribute(model, "INTPNT_CO_TOL_MU_RED", 1e-4)
        # else
        #     set_objective_function(model, -tr(W_hat))
        # end

        model = build_model(safe_bounds, Z_k, X′, A_hat, B_hat, n, m, n_t, t_horizon, Z_cur, problem.𝒩.Σ, problem.𝒩.Σ, scaler, delta_maxs, method, sim, problem.equal_time_constraint)

        # Convergence check
        if abs(obj - current_obj) < tol
            println("Converged after $iter iterations")
            break
        end
        obj = current_obj

    end

    end_time = time()
    println("Time spent in plan_control_inputs: ", end_time - plan_time)
    if sim == "xplane"
        return Z_cur, false
    else
        return Z_cur
    end
end



