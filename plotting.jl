using Plots
using Measures 
using StatsBase

function Plots.plot(problem::InputOptimizationProblem, Z_planned::Matrix{Float64})
    Z = problem.Z
    times = problem.times
    Z_unscaled = StatsBase.reconstruct(problem.scaler, Z)
    Z_planned_unscaled = StatsBase.reconstruct(problem.scaler, Z_planned)

    execution_times = times[end] .+ problem.Δt .* collect(1:problem.t_horizon)
    t_data = length(times)+1

    plts = []
    labels = ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow"]
    for i in 1:length(labels)
        if i in collect(2:9)
            # convert rad to deg
            plot(times, rad2deg(Z_unscaled[i, :]), label=labels[i])
            plt = plot!(execution_times, rad2deg(Z_planned_unscaled[i, t_data:end]), label=labels[i], linestyle=:dash)
        else
            plt = plot(times, Z_unscaled[i, :], label=labels[i])
            plot!(execution_times, Z_planned_unscaled[i, t_data:end], label=labels[i], linestyle=:dash)
        end
        push!(plts, plt)
    end
    plt = plot(plts..., layout=(7, 2), size=(800, 800), margin=5mm)
    savefig("/Users/joshuaott/Downloads/plot.pdf")

    # plot controls 
    plts = []
    labels = ["throt", "ele", "ail", "rud"]
    for i in 14:17
        if i > 14
            plot(times, rad2deg(Z_unscaled[i, :]), label=labels[i-13])
            plt = plot!(execution_times, rad2deg(Z_planned_unscaled[i, t_data:end]), label=labels[i-13], linestyle=:dash)
        else
            plot(times, Z_unscaled[i, :], label=labels[i-13])
            plt = plot!(execution_times, Z_planned_unscaled[i, t_data:end], label=labels[i-13], linestyle=:dash)
        end
        push!(plts, plt)
    end
    plt = plot(plts..., layout=(4, 1), size=(800, 800), margin=5mm)
    savefig("/Users/joshuaott/Downloads/plot_controls.pdf")
end
