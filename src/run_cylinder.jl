using WaterLily
using StaticArrays
using Plots; gr()
using Interpolations


function flood(f::Array; shift=(0.,0.), cfill=:RdBu_11, clims=(), levels=50, kv...)
    if length(clims) == 2
        @assert clims[1] < clims[2]
        @. f = min(clims[2], max(clims[1], f))
    else
        clims = (minimum(f), maximum(f))
    end
    Plots.contourf(axes(f,1) .+ shift[1], axes(f,2) .+ shift[2], f',
        linewidth=0, levels=levels, color=cfill, clims=clims, 
        aspect_ratio=:equal, ticks=false, grid=false; kv...)
end

addbody(x, y; c=:black) = Plots.plot!(Plots.Shape(x, y), c=c, legend=false)

function body_plot!(sim; levels=[0], lines=:black, R=inside(sim.flow.p))
    WaterLily.measure_sdf!(sim.flow.σ, sim.body, WaterLily.time(sim))
    contour!(sim.flow.σ[R]' |> Array; levels, lines)
end

function plot_rotating_body!(center, radius, θ)
    # Draw the circle
    n_points = 100
    angles = range(0, stop=2π, length=n_points)
    x_circle = center[1] .+ radius .* cos.(angles)
    y_circle = center[2] .+ radius .* sin.(angles)
    addbody(x_circle, y_circle)

    # Draw the line
    x_line = [center[1], center[1] + radius * cos(θ)]
    y_line = [center[2], center[2] + radius * sin(θ)]
    plot!(x_line, y_line, c=:white, lw=2)
end

function sim_gif!(sim, rotation_function; duration=1, step=0.1, verbose=true, R=inside(sim.flow.p),
                  remeasure=true, plotbody=false, center, radius, kv...)
    t₀ = round(sim_time(sim))
    sigmas = []
    times = []
    ωs = []
    @time @gif for tᵢ in range(t₀, t₀ + duration; step)
        sim_step!(sim, tᵢ; remeasure)
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
        sigma = deepcopy(sim.flow.σ[R] |> Array)
        flood(sim.flow.σ[R] |> Array; kv...)
        if plotbody
            body_plot!(sim)
            θ = rotation_function(tᵢ)
            plot_rotating_body!(center, radius, -θ)
        end

        verbose && println("tU/L=", round(tᵢ, digits=4),
            ", Δt=", round(sim.flow.Δt[end], digits=3))
        
        # Save the vorticity field
        push!(sigmas, sigma)
        push!(times, tᵢ)
        push!(ωs, rotation_function(tᵢ))
    end
    return sigmas, times, ωs
end

function sim!(sim, rotation_function; duration=1, step=0.1, verbose=true, R=inside(sim.flow.p),
                  remeasure=true, plotbody=false, center, radius, kv...)
    t₀ = round(sim_time(sim))
    sigmas = []
    times = []
    ωs = []
    for tᵢ in range(t₀, t₀ + duration; step)
        sim_step!(sim, tᵢ; remeasure)
        @inside sim.flow.σ[I] = WaterLily.curl(3, I, sim.flow.u) * sim.L / sim.U
        sigma = deepcopy(sim.flow.σ[R] |> Array)

        verbose && println("tU/L=", round(tᵢ, digits=4),
            ", Δt=", round(sim.flow.Δt[end], digits=3))
        
        # Save the vorticity field
        push!(sigmas, sigma)
        push!(times, tᵢ)
        push!(ωs, rotation_function(tᵢ))
    end
    return sigmas, times, ωs
end

function rotating_circle(n, m, rotation_function; Re=100, U=1)
    # Define a circle at the domain center
    radius = m / 8
    center = SA[50, m/2]

    function sdf(x,t)
        √sum(abs2, x .- center) - radius
    end

    function map(x,t)
        θ = rotation_function(t*U/radius)
        R = SMatrix{2, 2}([cos(θ) -sin(θ); sin(θ) cos(θ)])
        R * (x .- center) .+ center
    end

    # Create the simulation
    ν = U * radius / Re
    sim = Simulation((n, m), (U, 0), radius; ν, body=AutoBody(sdf,map))
    return sim, center, radius
end


function rotation_angle(t)
    return 8*π*sin(0.125*t)
end

function run_cylinder(;duration=200)
    # Run the simulation and create the gif
    U = 1.0
    sim, center, radius = rotating_circle(320, 160, rotation_angle, U=U)
    sigmas, times, ωs = sim_gif!(sim, rotation_angle, center=center, radius=radius, duration=duration, step=0.1, clims=(-8, 8), plotbody=true)

    return sigmas, times, ωs
end

function run_cylinder_planned_inputs(controls)
    U = 1.0
    step = 0.1
    duration = (length(controls)-1)*step
    ts = 0.0:step:duration
    interp = LinearInterpolation(ts, controls, extrapolation_bc=Line())
    function custom_rotation(t)
        return interp(t)
    end

    # Run the simulation and create the gif
    sim, center, radius = rotating_circle(320, 160, custom_rotation, U=U)
    sigmas, times, ωs = sim!(sim, custom_rotation, center=center, radius=radius, duration=duration, step=step, clims=(-8, 8), plotbody=true)

    return sigmas, times, ωs
end

