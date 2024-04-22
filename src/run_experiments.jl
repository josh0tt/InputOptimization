using JLD2
using ProgressMeter

# include("InputOptimization.jl")

mutable struct SimData
    problem::InputOptimizationProblem
    scaled_objectives::Vector{Float64}
    unscaled_objectives::Vector{Float64}
    runtimes::Vector{Float64}
    Zs::Vector{Matrix{Float64}}
end

"""
    compute_sigma(Z_t::Matrix{Float64}, n::Int64)

Compute the standard deviation of the residual in a linear system.

# Arguments
- `Z::Matrix{Float64}`: The matrix of observed data.
- `n::Int64`: The number of states.

# Returns
- `sigma`: The standard deviation of the residual.
"""
function compute_sigma(Z::Matrix{Float64}, n::Int64)
    p = size(Z, 2) - 1
    X_prime = Z[1:n, 2:end]
    X = Z[1:n, 1:p]
    U = Z[n+1:end, 1:p]

    # Solve the linear system using pseudo-inverse
    Theta = X_prime * pinv(Z[:, 1:p])

    A = Theta[:, 1:n]
    B = Theta[:, n+1:end]

    residual = X_prime - (A * X + B * U)
    residual_norm = norm(residual, 2)
    sigma = std(residual)

    return sigma, residual_norm
end

function compute_objective(Z::Matrix{Float64}, n::Int64)
    sigma, residual_norm = compute_sigma(Z, n)
    obj = tr(sigma^(2) * inv(Z * Z'))
    # obj = log(det(sigma^(2) * inv(Z[:, 1:i-1] * Z[:, 1:i-1]')))
    return obj
end

function run_experiments()
    problem = problem_setup()

    ccp_data = SimData(problem, Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Matrix{Float64}}())
    ccp_sdp_data = SimData(problem, Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Matrix{Float64}}())
    orthog_data = SimData(problem, Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Matrix{Float64}}())
    random_data = SimData(problem, Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Matrix{Float64}}())

    # run each method once first to compile
    solve(problem, ConvexConcave());
    solve(problem, ConvexConcaveSDP());
    solve(problem, OrthogonalMultisine());
    solve(problem, RandomSequence());

    num_sims = 100

    @showprogress dt=0.5 desc="Running sims..." for i in 1:num_sims
        for method in [ConvexConcave(), ConvexConcaveSDP(), OrthogonalMultisine(), RandomSequence()]
        # for method in [ConvexConcaveSDP()]
            Z_planned, runtime, times_actual, Z_actual = nothing, nothing, nothing, nothing

            attempts = 0
            max_attempts = 3
            success = false

            while attempts < max_attempts && !success
                try
                    Z_planned, runtime = @timed solve(problem, method)
                    times_actual, Z_actual = run_f16_sim(problem, Z_planned)
                    success = true  # Mark as successful if all commands execute without error
                catch e
                    attempts += 1
                    println("Caught error on attempt $attempts")
                    if attempts == max_attempts
                        return @error("Failed after $max_attempts attempts")
                    end
                end
            end
            
            Z_final = hcat(problem.Z, Z_actual[:, 2:end]);
            Z_final_unscaled = StatsBase.reconstruct(problem.scaler, Z_final)
            Z_actual_unscaled = StatsBase.reconstruct(problem.scaler, Z_actual)

            if method == ConvexConcave()
                data = ccp_data
            elseif method == ConvexConcaveSDP()
                data = ccp_sdp_data
            elseif method == OrthogonalMultisine()
                data = orthog_data
            else
                data = random_data
            end

            data.scaled_objectives = push!(data.scaled_objectives, compute_objective(Z_final, problem.n))
            data.unscaled_objectives = push!(data.unscaled_objectives, compute_objective(Z_final_unscaled, problem.n))
            data.runtimes = push!(data.runtimes, runtime)
            data.Zs = push!(data.Zs, Z_actual_unscaled)
        end

        sleep(0.1)
    end

    JLD2.save("ccp_sdp_data.jld2", "ccp_sdp_data", ccp_sdp_data)
    JLD2.save("ccp_data.jld2", "ccp_data", ccp_data)
    JLD2.save("orthog_data.jld2", "orthog_data", orthog_data)
    JLD2.save("random_data.jld2", "random_data", random_data)
end