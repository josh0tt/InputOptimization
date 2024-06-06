using NLopt
using Plots
using Statistics
using LinearAlgebra

# rms function
function rms(x)
    return sqrt(mean(x .^ 2))
end

function generateSignal(ts::Vector{Float64}, n_i::Int64, t_horizon::Int64, M::Int64, A::Vector{Float64}, phi::Matrix{Float64}, ωs::Matrix{Float64}, P::Matrix{Float64})
    U = zeros(Float64, t_horizon, n_i)
    for j = 1:n_i
        A_j = A[j]
        U[:, j] = sum(A_j * sqrt(P[j,k]) * sin.(ωs[j, k] * ts .+ phi[j, k]) for k = 1:M)
    end
    return U
end

function objectiveFunction(x::Vector, grad::Vector, data::Tuple{Int64, Int64, Int64, Vector{Float64}, Float64, Matrix{Float64}, Matrix{Float64}})
    n_i, M, n_t, ts, T, ωs, P = data
    A = x[1:n_i]
    phi = reshape(x[n_i+1:end], n_i, M)
    cost = 0.0
    for j = 1:n_i
        U = sum(A[j] * sqrt(P[j,k]) * sin.(ωs[j, k] * ts .+ phi[j, k]) for k in 1:M)
        cost += (maximum(U) - minimum(U)) / (2*sqrt(2)*rms(U))
    end
    return cost
end

function optimize_signal(x0::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, data::Tuple{Int64, Int64, Int64, Vector{Float64}, Float64, Matrix{Float64}, Matrix{Float64}})
    # opt = Opt(:LD_SLSQP, length(x0))
    opt = Opt(:LN_NELDERMEAD, length(x0))
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.min_objective = (x, grad) -> objectiveFunction(x, grad, data)
    opt.xtol_rel = 1e-6
    opt.maxeval = 100000  # Increase if necessary

    (minf, x_opt, ret) = optimize(opt, x0)
    numevals = opt.numevals # the number of function evaluations
    println("got $minf after $numevals iterations (returned $ret)")

    return x_opt, minf, ret
end

function findTimeShift(U, ts, U_ref)
    @show U_ref
    n_i = size(U, 2)
    shift_idx = zeros(Int, n_i)
    for i in 1:n_i
        min_i = argmin((U[:,i] .- U_ref[i]).^2)
        shift_idx[i] = min_i
    end
    taus = ts[shift_idx]
    return taus
end

function adjustPhaseForAllInputs(A, phi, taus, T, M, n_i, ωs)
    phi_adjusted = copy(phi)  # Create a copy to avoid modifying the original

    for j = 1:n_i  # For each control effector
        for k = 1:M  # For each sinusoidal component
            ωjk = ωs[j, k]  # Angular frequency for this component
            Δφjk = ωjk * taus[j]  # Phase offset for this component
            phi_adjusted[j, k] += Δφjk
        end
    end

    return phi_adjusted
end

# Compute RPF for each input
function calculateRPF(U, n_i)
    rpf = zeros(n_i)
    for j = 1:n_i
        peak_to_peak = maximum(U[:, j]) - minimum(U[:, j])
        rpf[j] = peak_to_peak / (2*sqrt(2) * rms(U[:, j]))
    end
    return rpf
end

function run_orthogonal_multisines(problem::InputOptimizationProblem)
    # ts = problem.times
    ts = problem.Δt .* collect(1:problem.t_horizon)
    T = problem.t_horizon * problem.Δt
    n_t = problem.n_t
    n_i = problem.m
    t_horizon = problem.t_horizon

    n = problem.n
    m = problem.m
    Z = problem.Z
    A_hat = problem.A_hat
    B_hat = problem.B_hat
    max_As = problem.max_As
    if problem.row_names[1] != "cylinder"
        Z_unscaled = StatsBase.reconstruct(problem.scaler, Z)
        U_desired = Z_unscaled[n+1:end, end] # last control input (current), where we want to start at 
    else
        U_desired = Z[n+1:end, end] # last control input (current), where we want to start at
    end

    M = 14 # Number of sinusoidal components

    A0 = max_As .* rand(problem.rng, n_i, 1)
    phi0 = rand(problem.rng, n_i, M)* π #* 2 * π
    x0 = vcat(A0[:], phi0[:])
    lb = [zeros(length(A0[:])); zeros(length(phi0[:]))]
    # ub = [max_A .* ones(length(A0[:])); 2*π*ones(length(phi0[:]))]
    ub = [max_As; 2*π*ones(length(phi0[:]))]

    # Define the angular frequencies from Morelli 2021
    f_min, f_max = problem.f_min, problem.f_max
    fs = LinRange(f_min, f_max, n_i*M)
    fs = reshape(fs, n_i, M)
    # convert to integers required for the same base period for ω = 2πk/T
    ks = round.(Int, fs * T)
    ωs = 2*π*ks / T

    # Initialize power fraction matrix P with zeros for 3 controls and M frequencies
    if problem.row_names[1] != "cylinder"
        P = zeros(4, M)

        # Assign power fractions based on the bar chart for each control from Morelli 2021
        P[1, :] = [0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04] 
        P[2, :] = [0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04] 
        P[3, :] = [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
        P[4, :] = [0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04]
    else
        P = zeros(n_i, M)
        # Assign power fractions based on the bar chart for each control from Morelli 2021
        P[1, :] = [0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04] 
    end 
    # P[1, :] = [0.1, 0.3, 0.3, 0.2, 0.1] 
    # P[2, :] = [0.1, 0.3, 0.3, 0.2, 0.1]
    # P[3, :] = [0.1, 0.3, 0.3, 0.2, 0.1]
    # P[4, :] = [0.1, 0.3, 0.3, 0.2, 0.1]  

    # Perform optimization
    data = (n_i, M, n_t, ts, T, ωs, P)
    x_opt, minf, ret = optimize_signal(x0, lb, ub, data)

    # Post-optimization: Retrieve optimized variables and reconstruct U
    A_opt = x_opt[1:n_i] #reshape(x_opt[1:n_i], n_i, 1) #.* sqrt(1/M)
    phi_opt = reshape(x_opt[n_i+1:end], n_i, M)
    U_opt = generateSignal(ts, n_i, t_horizon, M, A_opt, phi_opt, ωs, P)

    taus = findTimeShift(U_opt, ts, zeros(n_i))

    # Adjust the phase angles based on the calculated tau
    phi_opt_adjusted = adjustPhaseForAllInputs(A_opt, phi_opt, taus, T, M, n_i, ωs)

    # Generate the adjusted signal
    U_adjusted = generateSignal(ts, n_i, t_horizon, M, A_opt, phi_opt_adjusted, ωs, P)

    @show U_adjusted[end,:]

    rpf = calculateRPF(U_adjusted, n_i)

    U_adjusted .+= U_desired'

    println(ret)

    control_traj = U_adjusted'
    if problem.row_names[1] != "cylinder"
        control_traj = StatsBase.transform(problem.scaler, vcat(zeros(n, t_horizon), control_traj))
        control_traj = control_traj[n+1:end, :]
    end
    Z_planned = zeros(n+m, t_horizon+n_t)
    Z_planned[:, 1:n_t] = Z
    Z_planned[n+1:end, n_t+1:end] = control_traj[1:m, :]

    for i in n_t:(n_t+t_horizon-1)
        Z_planned[1:n, i+1] .= A_hat * Z_planned[1:n, i] + B_hat*Z_planned[n+1:end, i]
        # Z_planned[1:n, n_t + i] = Z_planned[1:n, n_t + i - 1] + A_hat * Z_planned[1:n, n_t + i - 1] + B_hat * control_traj[i, :]
    end

    return Z_planned
    # return U_adjusted, A_opt, phi_opt_adjusted, ts, T, M, n_i, ωs, P
end