using Plots
using Measures 
using StatsBase

function Plots.plot(problem::InputOptimizationProblem, Z_planned::Matrix{Float64})
    Z = problem.Z
    n_t = problem.n_t
    n = problem.n
    m = problem.m
    times = problem.times
    Z_unscaled = StatsBase.reconstruct(problem.scaler, Z)
    Z_planned_unscaled = StatsBase.reconstruct(problem.scaler, Z_planned)

    execution_times = times[end] .+ problem.Δt .* collect(1:problem.t_horizon)

    plts = []
    labels = ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow"]
    for i in 1:length(labels)
        plt = plot(times, Z_unscaled[i, :], label=labels[i])
        plot!(execution_times, Z_planned_unscaled[i, n_t+1:end], label=labels[i], linestyle=:dash)
        push!(plts, plt)
    end
    plt = plot(plts..., layout=(7, 2), size=(800, 800), margin=5mm)
    savefig("/Users/joshuaott/Downloads/plot.pdf")

    # plot controls 
    plts = []
    labels = ["throt", "ele", "ail", "rud"]
    for i in n+1:n+m
        plot(times, Z_unscaled[i, :], label=labels[i-13])
        plt = plot!(execution_times, Z_planned_unscaled[i, n_t+1:end], label=labels[i-13], linestyle=:dash)
        push!(plts, plt)
    end
    plt = plot(plts..., layout=(4, 1), size=(800, 800), margin=5mm)
    savefig("/Users/joshuaott/Downloads/plot_controls.pdf")
end


function Plots.plot(problem::InputOptimizationProblem, Z_planned::Matrix{Float64}, Z_actual::Matrix{Float64}, times_actual::Vector{Float64})
    Z = problem.Z
    n_t = problem.n_t
    n = problem.n
    m = problem.m
    times = problem.times
    t_horizon = problem.t_horizon
    A_hat = problem.A_hat
    B_hat = problem.B_hat
    safe_bounds_unscaled = problem.safe_bounds_unscaled

    Z_unscaled = StatsBase.reconstruct(problem.scaler, Z)
    Z_planned_unscaled = StatsBase.reconstruct(problem.scaler, Z_planned)
    Z_actual_unscaled = StatsBase.reconstruct(problem.scaler, Z_actual)

    execution_times = times[end] .+ problem.Δt .* collect(1:problem.t_horizon)

    println("Max control diff: ", maximum(abs.(Z_actual_unscaled[n+1:end, 2:end] .- Z_actual_unscaled[n+1:end, 1:end-1]), dims=2))

    # create predicted Z. This is the output that would we would predict given the control inputs that were actually executed from the autopilot.
    Z_predicted = zeros(n+m, t_horizon)
    Z_predicted[:, 1] = Z_actual[:, 1]
    Z_predicted[n+1:end, :] = Z_actual[n+1:end, 1:size(Z_predicted, 2)]

    for i in 1:(t_horizon-1)
        Z_predicted[1:n, i+1] = A_hat * Z_predicted[1:n, i] + B_hat * Z_predicted[n+1:end, i]
    end
    Z_predicted_unscaled = StatsBase.reconstruct(problem.scaler, Z_predicted)

    plts = []
    labels = ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow"]
    for i in 1:length(labels)
        plot(times, Z_unscaled[i, :], label=labels[i], title=labels[i])
        plot!(execution_times, Z_planned_unscaled[i, n_t+1:end], label="planned", linestyle=:dash)
        # plot!(times_actual, Z_predicted_unscaled[i, :], label="predicted", linestyle=:dot)
        # plot safe bounds as dashed horizontal lines 
        plot!([times[1], times_actual[end]], [safe_bounds_unscaled[i, 1], safe_bounds_unscaled[i, 1]], label="lower bound", linestyle=:dash, color=:black)
        plot!([times[1], times_actual[end]], [safe_bounds_unscaled[i, 2], safe_bounds_unscaled[i, 2]], label="upper bound", linestyle=:dash, color=:black)
        plt = plot!(times_actual, Z_actual_unscaled[i, :], label="actual", linestyle=:dot)
        push!(plts, plt)
    end
    plt = plot(plts..., layout=(7, 2), size=(800, 800), margin=5mm, legend=false)
    savefig("/Users/joshuaott/Downloads/plot.pdf")

    # plot controls 
    plts = []
    labels = ["throt", "ele", "ail", "rud"]
    for i in n+1:n+m
        plot(times, Z_unscaled[i, :], label=labels[i-13])
        plot!(execution_times, Z_planned_unscaled[i, n_t+1:end], label=labels[i-13], linestyle=:dash)
        # plot safe bounds as dashed horizontal lines
        plot!([times[1], times_actual[end]], [safe_bounds_unscaled[i, 1], safe_bounds_unscaled[i, 1]], label="lower bound", linestyle=:dash, color=:black)
        plot!([times[1], times_actual[end]], [safe_bounds_unscaled[i, 2], safe_bounds_unscaled[i, 2]], label="upper bound", linestyle=:dash, color=:black)
        plt = plot!(times_actual, Z_actual_unscaled[i, :], label=labels[i-13], linestyle=:dot)
        push!(plts, plt)
    end
    plt = plot(plts..., layout=(4, 1), size=(800, 800), margin=5mm, legend=false)
    savefig("/Users/joshuaott/Downloads/plot_controls.pdf")
end