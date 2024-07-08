using JLD2

struct CylinderFlowData
    sigmas::Array{Array{Float64,2},1}
    times::Array{Float64,1}
    ωs::Array{Float64,1}
    A_hat::Matrix{Float64}
    B_hat::Matrix{Float64}
    ϕ::AbstractMatrix                                   
    W::AbstractMatrix
    transform::AbstractMatrix
    U_hat::Matrix{Float64}
end

"""
    find_max_As(n::Int, m::Int, Z_unscaled::Matrix{Float64}, safe_bounds_unscaled::Matrix{Float64})

Given the number of states `n`, the number of control inputs `m`, the unscaled state trajectory matrix `Z_unscaled`, and the unscaled safe bounds matrix `safe_bounds_unscaled`, this function calculates the maximum allowable deviation (amplitude) from the desired control input for each control input.

# Arguments
- `n::Int`: The number of states.
- `m::Int`: The number of control inputs.
- `Z_unscaled::Matrix{Float64}`: The unscaled state trajectory matrix.
- `safe_bounds_unscaled::Matrix{Float64}`: The unscaled safe bounds matrix.

# Returns
- `max_As::Vector{Float64}`: A vector containing the maximum allowable deviation (amplitude) from the desired control input for each control input.

"""
function find_max_As(n::Int, m::Int, Z_unscaled::Matrix{Float64}, safe_bounds_unscaled::Matrix{Float64})
    U_desired = Z_unscaled[n+1:end, end] # last control input (current), where we want to start at 
    max_As = [minimum([abs(U_desired[i-n] - safe_bounds_unscaled[i, 1]), abs(U_desired[i-n] - safe_bounds_unscaled[i, 2])]) for i in n+1:n+m]
    return max_As
end

"""
    fit_process_noise(Z::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int, m::Int)

Fit process noise using maximum likelihood estimation.

# Arguments
- `Z::Matrix{Float64}`: The input matrix containing the state, control, and next state observations.
- `A_hat::Matrix{Float64}`: The estimated state transition matrix.
- `B_hat::Matrix{Float64}`: The estimated control matrix.
- `n::Int`: The number of state variables.
- `m::Int`: The number of control variables.

# Returns
- `𝒩`: The fitted process noise distribution.

"""
function fit_process_noise(Z::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int, m::Int)::FullNormal
    x = Z[1:n, 1:end-1]
    u = Z[n+1:end, 1:end-1]
    xp = Z[1:n, 2:end]

    𝒩 = fit_mle(MvNormal, xp - A_hat*x - B_hat*u)

    return 𝒩
end

"""
f16_problem_setup()

This function sets up the problem for input optimization in the F16 dynamics case. It performs the following steps:

1. Run F16 waypoint simulation to collect a data set.
2. Scale the data using a unit range transform.
3. Estimate the linear system using the scaled data.
4. Create an `InputOptimizationProblem` object.

The function returns the `InputOptimizationProblem` object.

# Arguments
- None

# Returns
- `problem`: An `InputOptimizationProblem` object representing the input optimization problem.

"""
function f16_problem_setup()::InputOptimizationProblem
    rng = MersenneTwister(123456789)

    # 1. run F16 waypoint simulation to collect data set 
    # we don't need it this smooth, we should just limit how far from the starting init we can go
    times, states, controls = run_f16_waypoint_sim()
    n, m = size(states, 2), size(controls, 2)
    n_t = length(times)
    Δt = times[2] - times[1]
    t_horizon = round(Int64, 7.0 / Δt)

    # 2. scale the data
    Z_unscaled = Matrix(hcat(states, controls)') # Z is shaped as (n+m,t) where n is the number of states and m is the number of controls    
    scaler = fit(UnitRangeTransform, Z_unscaled, dims=2)
    Z = StatsBase.transform(scaler, Z_unscaled)

    safe_bounds_unscaled = [
        Z_unscaled[1, end]-150 Z_unscaled[1, end]+150; # vt ft/s
        Z_unscaled[2, end]-10 Z_unscaled[2, end]+20; # alpha
        Z_unscaled[3, end]-10 Z_unscaled[3, end]+10; # beta
        Z_unscaled[4, end]-30 Z_unscaled[4, end]+30; # phi (roll)
        Z_unscaled[5, end]-30 Z_unscaled[5, end]+30; # theta (pitch)
        -180 180; # psi
        Z_unscaled[7, end]-30 Z_unscaled[7, end]+30; # P
        Z_unscaled[8, end]-30 Z_unscaled[8, end]+30; # Q
        Z_unscaled[9, end]-30 Z_unscaled[9, end]+30; # R
        -Inf Inf; # pn ft
        -Inf Inf; # pe ft
        Z_unscaled[12, end]-2000 Z_unscaled[12, end]+2000; # h ft
        0 100; # pow
        # -0.5 3; # Nz
        # minimum(Z_unscaled[14, :]) maximum(Z_unscaled[14, :]); # throt
        # the bounds that worked for large scale:
        # 0 1.0; # throt
        # minimum(Z_unscaled[15, :]) maximum(Z_unscaled[15, :]); # ele
        # minimum(Z_unscaled[16, :]) maximum(Z_unscaled[16, :]); # ail
        # minimum(Z_unscaled[17, :]) maximum(Z_unscaled[17, :]); # rud
        # Z_unscaled[14, end]-0.1 Z_unscaled[14, end]+0.5; # throt
        # Z_unscaled[15, end]-2.0 Z_unscaled[15, end]+2.0; # ele
        # Z_unscaled[16, end]-2.0 Z_unscaled[16, end]+2.0; # ail
        # Z_unscaled[17, end]-2.0 Z_unscaled[17, end]+2.0 # rud
        ########################################################
        0 Z_unscaled[14, end]+0.5; # throt
        Z_unscaled[15, end]-1 Z_unscaled[15, end]+1; # ele
        Z_unscaled[16, end]-1 Z_unscaled[16, end]+1; # ail
        Z_unscaled[17, end]-1 Z_unscaled[17, end]+1 # rud
        ########################################################
        # 0 Z_unscaled[14, end]+0.1; # throt
        # Z_unscaled[15, end]-0.5 Z_unscaled[15, end]+0.5; # ele
        # Z_unscaled[16, end]-0.5 Z_unscaled[16, end]+0.5; # ail
        # Z_unscaled[17, end]-0.5 Z_unscaled[17, end]+0.5 # rud
    ]
    
    # equating constraints between ccp and orthogonal multisines
    max_As = find_max_As(n, m, Z_unscaled, safe_bounds_unscaled)
    # Update the safe bounds to account for the maximum allowable deviation in control inputs
    for i in 1:m
        safe_bounds_unscaled[n+i, 1] = Z_unscaled[n+i, end] - max_As[i]
        safe_bounds_unscaled[n+i, 2] = Z_unscaled[n+i, end] + max_As[i]
    end
    f_min, f_max = 0.1, 1.7 # 0.2, 1.1 # Hz
    # create m sinuoids using max_As and ω=2*π*f_max
    sines = max_As .* sin.(2*π*f_max .* times)' # should be m x n_t
    sines_scaled = StatsBase.transform(scaler, vcat(zeros(n, n_t), sines))
    sines_scaled = sines_scaled[n+1:end, :]
    delta_maxs = [mean(abs.(sines_scaled[i, 2:end] - sines_scaled[i, 1:end-1])) for i in 1:m] #[maximum(abs.(sines_scaled[i, 2:end] - sines_scaled[i, 1:end-1])) for i in 1:m]
    @show delta_maxs

    # scale the bounds as well
    lower_bounds, upper_bounds = scale_bounds(scaler, safe_bounds_unscaled, 1, n + m)
    safe_bounds = zeros(size(safe_bounds_unscaled))
    for i in 1:size(safe_bounds, 1)
        safe_bounds[i, 1] = lower_bounds[i]
        safe_bounds[i, 2] = upper_bounds[i]
    end

    # 3. Dynamic Mode Decomposition with control
    # DMDc not needed for F16 state space of 13 states and 4 controls
    dimensionality_reduction = false 
    if dimensionality_reduction
        Ω = Z[:, 1:end-1]
        Xp = Z[1:n, 2:end]
        A_hat, B_hat, ϕ, W, transform, U_hat = DMDc(Ω, Xp)
        # now convert the Z data to the new basis
        Z_full = deepcopy(Z)
        Z = vcat(project_down(Z[1:n, :], U_hat), Z[n+1:end, :]) # control inputs u are not transformed 
        # update the safe bounds
        safe_bounds = vcat(project_down(safe_bounds[1:n, :], U_hat), safe_bounds[n+1:end, :])
        n = size(A_hat, 1)
        m = size(B_hat, 2)
        𝒩 = fit_process_noise(Z, A_hat, B_hat, n, m)
    else
        # estimate the linear system
        A_hat, B_hat = estimate_linear_system(Z, n)
        Z_full = deepcopy(Z)
        𝒩 = fit_process_noise(Z, A_hat, B_hat, n, m)
        ϕ = zeros(n, n_t)
        W = zeros(n, n_t)
        transform = zeros(n, n)
        U_hat = zeros(n, n)
    end

    # 4. create the InputOptimizationProblem
    X′ = Z[1:n, 2:end]
    Z = Z[:, 1:end-1]
    times = times[1:end-1]
    n_t = length(times)
    problem = InputOptimizationProblem(rng, Z, X′, scaler, times, A_hat, B_hat, 𝒩, n, m, n_t, t_horizon, Δt, safe_bounds, safe_bounds_unscaled, delta_maxs, max_As, f_min, f_max, ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow", "throt", "ele", "ail", "rud"], false)

    return problem
end

function f16_ground_truth_setup()::InputOptimizationProblem
    rng = MersenneTwister(123456)

    # 1. run F16 waypoint simulation to collect data set 
    # we don't need it this smooth, we should just limit how far from the starting init we can go
    times, states, controls = run_f16_ground_truth_waypoint_sim()
    n, m = size(states, 2), size(controls, 2)
    n_t = length(times)
    Δt = times[2] - times[1]
    t_horizon = round(Int64, 7 / Δt)

    # 2. scale the data
    Z_unscaled = Matrix(hcat(states, controls)') # Z is shaped as (n+m,t) where n is the number of states and m is the number of controls
    scaler = fit(UnitRangeTransform, Z_unscaled, dims=2)
    Z = StatsBase.transform(scaler, Z_unscaled)

    safe_bounds_unscaled = [
        Z_unscaled[1, end]-150 Z_unscaled[1, end]+150; # vt ft/s
        Z_unscaled[2, end]-10 Z_unscaled[2, end]+20; # alpha
        Z_unscaled[3, end]-10 Z_unscaled[3, end]+10; # beta
        Z_unscaled[4, end]-30 Z_unscaled[4, end]+30; # phi (roll)
        Z_unscaled[5, end]-30 Z_unscaled[5, end]+30; # theta (pitch)
        -180 180; # psi
        Z_unscaled[7, end]-30 Z_unscaled[7, end]+30; # P
        Z_unscaled[8, end]-30 Z_unscaled[8, end]+30; # Q
        Z_unscaled[9, end]-30 Z_unscaled[9, end]+30; # R
        -Inf Inf; # pn ft
        -Inf Inf; # pe ft
        Z_unscaled[12, end]-2000 Z_unscaled[12, end]+2000; # h ft
        0 100; # pow
        # -0.5 3; # Nz
        Z_unscaled[14, end]-0.1 Z_unscaled[14, end]+0.1; # throt
        Z_unscaled[15, end]-0.5 Z_unscaled[15, end]+0.5; # ele
        Z_unscaled[16, end]-0.5 Z_unscaled[16, end]+0.5; # ail
        Z_unscaled[17, end]-0.5 Z_unscaled[17, end]+0.5 # rud
    ]
    
    # equating constraints between ccp and orthogonal multisines
    max_As = find_max_As(n, m, Z_unscaled, safe_bounds_unscaled)
    # Update the safe bounds to account for the maximum allowable deviation in control inputs
    for i in 1:m
        safe_bounds_unscaled[n+i, 1] = Z_unscaled[n+i, end] - max_As[i]
        safe_bounds_unscaled[n+i, 2] = Z_unscaled[n+i, end] + max_As[i]
    end
    f_min, f_max = 0.1, 1.7 # 0.2, 1.1 # Hz
    # create m sinuoids using max_As and ω=2*π*f_max
    sines = max_As .* sin.(2*π*f_max .* times)' # should be m x n_t
    sines_scaled = StatsBase.transform(scaler, vcat(zeros(n, n_t), sines))
    sines_scaled = sines_scaled[n+1:end, :]
    delta_maxs = [mean(abs.(sines_scaled[i, 2:end] - sines_scaled[i, 1:end-1])) for i in 1:m] #[maximum(abs.(sines_scaled[i, 2:end] - sines_scaled[i, 1:end-1])) for i in 1:m]
    @show delta_maxs

    # scale the bounds as well
    lower_bounds, upper_bounds = scale_bounds(scaler, safe_bounds_unscaled, 1, n + m)
    safe_bounds = zeros(size(safe_bounds_unscaled))
    for i in 1:size(safe_bounds, 1)
        safe_bounds[i, 1] = lower_bounds[i]
        safe_bounds[i, 2] = upper_bounds[i]
    end

    # 3. Dynamic Mode Decomposition with control
    # DMDc not needed for F16 state space of 13 states and 4 controls
    dimensionality_reduction = false 
    if dimensionality_reduction
        Ω = Z[:, 1:end-1]
        Xp = Z[1:n, 2:end]
        A_hat, B_hat, ϕ, W, transform, U_hat = DMDc(Ω, Xp)
        # now convert the Z data to the new basis
        Z_full = deepcopy(Z)
        Z = vcat(project_down(Z[1:n, :], U_hat), Z[n+1:end, :]) # control inputs u are not transformed 
        # update the safe bounds
        safe_bounds = vcat(project_down(safe_bounds[1:n, :], U_hat), safe_bounds[n+1:end, :])
        n = size(A_hat, 1)
        m = size(B_hat, 2)
        𝒩 = fit_process_noise(Z, A_hat, B_hat, n, m)
    else
        # estimate the linear system
        A_hat, B_hat = estimate_linear_system(Z, n)
        Z_full = deepcopy(Z)
        𝒩 = fit_process_noise(Z, A_hat, B_hat, n, m)
        ϕ = zeros(n, n_t)
        W = zeros(n, n_t)
        transform = zeros(n, n)
        U_hat = zeros(n, n)
    end

    # 4. create the InputOptimizationProblem
    X′ = Z[1:n, 2:end]
    Z = Z[:, 1:end-1]
    times = times[1:end-1]
    n_t = length(times)
    problem = InputOptimizationProblem(rng, Z, X′, scaler, times, A_hat, B_hat, 𝒩, n, m, n_t, t_horizon, Δt, safe_bounds, safe_bounds_unscaled, delta_maxs, max_As, f_min, f_max, ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow", "throt", "ele", "ail", "rud"], false)

    return problem
end


function cylinder_problem_setup(;load_data=true, return_data=false)
    rng = MersenneTwister(123456)

    # 1. Load cylinder data 
    if load_data
        data_path = joinpath(@__DIR__, "..", "data", "cylinder", "cylinder_data.jld2")
        # data_path = joinpath(@__DIR__, "..", "data", "cylinder", "short_cylinder_data.jld2")
        cylinder_data = load(data_path)["cylinder_data"]
        sigmas, times, ωs = cylinder_data.sigmas, cylinder_data.times, cylinder_data.ωs

        # NOTE: these were constructed using DMDc AFTER normalization was applied 
        A_hat = cylinder_data.A_hat
        B_hat = cylinder_data.B_hat
        ϕ = cylinder_data.ϕ
        W = cylinder_data.W
        transform = cylinder_data.transform
        U_hat = cylinder_data.U_hat
    else
        sigmas, times, ωs = run_cylinder()
    end

    # 2. Organize data  
    states = Matrix(hcat([vec(sigmas[i]) for i in 1:length(sigmas)]...)')
    controls = ωs
    n, m = size(states, 2), size(controls, 2)
    n_t = length(times)
    Δt = times[2] - times[1]
    t_horizon = round(Int64, 100 / Δt)
    Z = Matrix{Float64}(hcat(states, controls)')   

    # 3. Normalize the data
    scaler = StatsBase.fit(UnitRangeTransform, Z, dims=2)
    Z = StatsBase.transform(scaler, Z)
    Z[isnan.(Z)] .= 0 # set NaNs to 0


    # 3. Dynamic Mode Decomposition with control
    if !load_data        
        Ω = Z[:, 1:end-1]
        Xp = Z[1:n, 2:end]
        A_hat, B_hat, ϕ, W, transform, U_hat = InputOptimization.DMDc(Ω, Xp, 0.9)
    end

    # now convert the Z data to the new basis
    Z_full = deepcopy(Z)
    Z = vcat(InputOptimization.project_down(Z[1:n, :], U_hat), Z[n+1:end, :]) # control inputs u are not transformed 
    n = size(A_hat, 1)
    m = size(B_hat, 2)
    𝒩 = InputOptimization.fit_process_noise(Z, A_hat, B_hat, n, m)

    # scaler = StatsBase.fit(UnitRangeTransform, Z_full, dims=2)
    safe_bounds = vcat([[-Inf, Inf] for i in 1:n], [[minimum((Z[n+1:end, :])), maximum(Z[n+1:end, :])] for i in 1:m])
    safe_bounds = Matrix(hcat(safe_bounds...)')
    safe_bounds_unscaled = safe_bounds
    max_As = [(maximum(Z[n+1:end, :]) - minimum(Z[n+1:end, :]))/2]
    f_min, f_max = 0.04, 0.1
    sines = max_As .* sin.(2*π*f_max .* times)' 
    delta_maxs = [mean(abs.(sines[i, 2:end] - sines[i, 1:end-1])) for i in 1:m]

    # 4. create the InputOptimizationProblem
    X′ = Z[1:n, 2:end]
    Z = Z[:, 1:end-1]
    times = times[1:end-1]
    sigmas = sigmas[1:end-1]
    ωs = ωs[1:end-1]
    n_t = length(times)
    problem = InputOptimizationProblem(rng, Z, X′, scaler, times, A_hat, B_hat, 𝒩, n, m, n_t, t_horizon, Δt, safe_bounds, safe_bounds_unscaled, delta_maxs, max_As, f_min, f_max, ["cylinder"], false)

    if return_data
        return problem, ϕ, W, transform, U_hat, sigmas, times, ωs
    else
        return problem
    end
end