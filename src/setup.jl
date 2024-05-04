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
function fit_process_noise(Z::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int, m::Int)
    x = Z[1:n, 1:end-1]
    u = Z[n+1:end, 1:end-1]
    xp = Z[1:n, 2:end]

    𝒩 = fit_mle(MvNormal, xp - A_hat*x - B_hat*u)

    return 𝒩
end

"""
    problem_setup()

This function sets up the problem for input optimization. It performs the following steps:

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
function problem_setup()
    rng = MersenneTwister(1234567)

    # 1. run F16 waypoint simulation to collect data set 
    # we don't need it this smooth, we should just limit how far from the starting init we can go
    times, states, controls = run_f16_waypoint_sim()
    n, m = size(states, 2), size(controls, 2)
    n_t = length(times)
    Δt = times[2] - times[1]
    t_horizon = round(Int64, 10 / Δt)

    # 2. scale the data
    Z_unscaled = Matrix(hcat(states, controls)') # Z is shaped as (n+m,t) where n is the number of states and m is the number of controls
    scaler = fit(UnitRangeTransform, Z_unscaled, dims=2)
    Z = StatsBase.transform(scaler, Z_unscaled)

    safe_bounds_unscaled = [
        Z_unscaled[1, end]-150 Z_unscaled[1, end]+150; # vt ft/s
        Z_unscaled[2, end]-10 Z_unscaled[2, end]+20; # alpha
        Z_unscaled[3, end]-2 Z_unscaled[3, end]+2; # beta
        Z_unscaled[4, end]-45 Z_unscaled[4, end]+45; # phi (roll)
        Z_unscaled[5, end]-60 Z_unscaled[5, end]+60; # theta (pitch)
        -180 180; # psi
        Z_unscaled[7, end]-30 Z_unscaled[7, end]+30; # P
        Z_unscaled[8, end]-30 Z_unscaled[8, end]+30; # Q
        Z_unscaled[9, end]-30 Z_unscaled[9, end]+30; # R
        -Inf Inf; # pn ft
        -Inf Inf; # pe ft
        Z_unscaled[12, end]-2000 Z_unscaled[12, end]+2000; # h ft
        0 100; # pow
        # -0.5 3; # Nz
        Z_unscaled[14, end]-0.1 Z_unscaled[14, end]+0.4; # throt
        Z_unscaled[15, end]-1 Z_unscaled[15, end]+1; # ele
        Z_unscaled[16, end]-1 Z_unscaled[16, end]+1; # ail
        Z_unscaled[17, end]-1 Z_unscaled[17, end]+1 # rud
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

    # 3. estimate the linear system
    A_hat, B_hat = estimate_linear_system(Z, n)
    𝒩 = fit_process_noise(Z, A_hat, B_hat, n, m)

    # 4. create the InputOptimizationProblem
    problem = InputOptimizationProblem(rng, Z, scaler, times, A_hat, B_hat, 𝒩, n, m, n_t, t_horizon, Δt, safe_bounds, safe_bounds_unscaled, delta_maxs, max_As, f_min, f_max, ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow", "throt", "ele", "ail", "rud"])

    return problem
end