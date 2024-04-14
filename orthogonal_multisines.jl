using NLopt
using Plots
using Statistics
using LinearAlgebra

# rms function
function rms(x)
    return sqrt(mean(x .^ 2))
end

function generateSignal(ts::Vector{Float64}, n_i::Int64, M::Int64, T::Float64, A::Vector{Float64}, phi::Matrix{Float64}, ωs::Matrix{Float64}, P::Matrix{Float64})
    n_t = length(ts)
    U = zeros(Float64, n_t, n_i)
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

function optimizeWithSQP(x0::Vector{Float64}, lb::Vector{Float64}, ub::Vector{Float64}, data::Tuple{Int64, Int64, Int64, Vector{Float64}, Float64, Matrix{Float64}, Matrix{Float64}})
    # opt = Opt(:LD_SLSQP, length(x0))
    opt = Opt(:LN_NELDERMEAD, length(x0))
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.min_objective = (x, grad) -> objectiveFunction(x, grad, data)
    opt.xtol_rel = 1e-6
    # opt.maxeval = 10000  # Increase if necessary

    (minf, x_opt, ret) = optimize(opt, x0)
    numevals = opt.numevals # the number of function evaluations
    println("got $minf after $numevals iterations (returned $ret)")

    return x_opt, minf, ret
end

function findTimeShift(U, ts)
    taus = zeros(size(U,2))
    for j in 1:size(U,2)
        for i in 2:length(ts)
            if U[i-1, j] * U[i, j] <= 0
                taus[j] = ts[i]
                break
            end
        end
    end
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
function calculateRPF(U, n_i, n_t)
    rpf = zeros(n_i)
    for j = 1:n_i
        peak_to_peak = maximum(U[:, j]) - minimum(U[:, j])
        rpf[j] = peak_to_peak / (2*sqrt(2) * rms(U[:, j]))
    end
    return rpf
end

function scale_column(column, new_min, new_max)
    A, B = minimum(column), maximum(column)
    ((column .- A) ./ (B - A)) .* (new_max - new_min) .+ new_min
end

function shift(U, U_desired, n_i)    
    shift_idx = zeros(Int, n_i)
    U_shift = zeros(size(U))
    for i in 1:n_i
        shift_idx[i] = argmin((U[:,i] .- U_desired[i]).^2)
        U_shift[:, i] = [U[shift_idx[i]:end, i]; U[1:shift_idx[i]-1, i]]
    end

    return U_shift
end

function run_orthogonal_multisines(ts, T, n_t, U_desired::Vector{Float64}, t_horizon::Int64, n_i::Int64 = 4, max_A::Float64=0.1, M::Int64 = 14)
    # Constants and initial conditions as in your Julia code
    # T = 27.0
    # n_t = 1000
    # M = 14
    # n_i = 3
    # T = t_horizon
    # n_t = t_horizon#25

    # ts = range(0, stop=T, length=n_t)

    A0 = max_A .* rand(n_i, 1)
    phi0 = rand(n_i, M)* π #* 2 * π
    x0 = vcat(A0[:], phi0[:])
    lb = [zeros(length(A0[:])); zeros(length(phi0[:]))]
    ub = [max_A .* ones(length(A0[:])); 2*π*ones(length(phi0[:]))]

    # Define the angular frequencies
    f_min, f_max = 0.1, 1.7
    fs = LinRange(0.1, 1.7, n_i*M)
    fs = reshape(fs, n_i, M)
    # convert to integers required for the same base period for ω = 2πk/T
    ks = round.(Int, fs * T)
    ωs = 2*π*ks / T

    # Initialize power fraction matrix P with zeros for 3 controls and M frequencies
    P = zeros(4, M)

    # Assign power fractions based on the bar chart for each control
    P[1, :] = [0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04] 
    P[2, :] = [0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04] 
    P[3, :] = [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
    P[4, :] = [0.06, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04] 


    # Perform optimization
    data = (n_i, M, n_t, ts, T, ωs, P)
    x_opt, minf, ret = optimizeWithSQP(x0, lb, ub, data)

    # Post-optimization: Retrieve optimized variables and reconstruct U
    A_opt = x_opt[1:n_i] #reshape(x_opt[1:n_i], n_i, 1) #.* sqrt(1/M)
    phi_opt = reshape(x_opt[n_i+1:end], n_i, M)
    U_opt = generateSignal(ts, n_i, M, T, A_opt, phi_opt, ωs, P)

    taus = findTimeShift(U_opt, ts)

    # Adjust the phase angles based on the calculated tau
    phi_opt_adjusted = adjustPhaseForAllInputs(A_opt, phi_opt, taus, T, M, n_i, ωs)

    # Generate the adjusted signal
    U_adjusted = generateSignal(ts, n_i, M, T, A_opt, phi_opt_adjusted, ωs, P)


    rpf = calculateRPF(U_adjusted, n_i, n_t)

    # U_scaled = zeros(size(U_opt))
    # U_scaled[:, 1] = scale_column(U_opt[:, 1], U_desired[1] - 0.1, U_desired[1] + 0.1) # Elevator
    # U_scaled[:, 2] = scale_column(U_opt[:, 2], U_desired[2] - 0.1, U_desired[2] + 0.1) # Aileron
    # U_scaled[:, 3] = scale_column(U_opt[:, 3], U_desired[3] - 0.1, U_desired[3] + 0.1) # Rudder
    # U_scaled[:, 4] = scale_column(U_opt[:, 4], U_desired[4] - 0.1, U_desired[4] + 0.1) # Throttle

    # U_shift = shift(U_scaled, U_desired, n_i)
    U_adjusted .+= U_desired'

    println(ret)

    return U_adjusted, A_opt, phi_opt_adjusted, ts, T, M, n_i, ωs, P
    # return U_shift, A_opt, phi_opt, ts, T, M, n_i, ωs, P
end