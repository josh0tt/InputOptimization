function run_f16_waypoint_sim()
    pushfirst!(PyVector(pyimport("sys")."path"), "")

    py"""
    import math
    import numpy as np
    from numpy import deg2rad

    from aerobench.run_f16_sim import run_f16_sim
    from aerobench.visualize import anim3d, plot
    from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot



    def simulate():
        'simulate the system, returning waypoints, res'

        ### Initial Conditions ###
        power = 9 # engine power level (0-10)

        # Default alpha & beta
        alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
        beta = 0                # Side slip angle (rad)

        # Initial Attitude
        alt = 3800        # altitude (ft)
        vt = 540          # initial velocity (ft/sec)
        phi = 0           # Roll angle from wings level (rad)
        theta = 0         # Pitch angle from nose level (rad)
        psi = math.pi/8   # Yaw angle from North (rad)

        # Build Initial Condition Vectors
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
        tmax = 70 # simulation time

        # make waypoint list
        e_pt = 1000
        n_pt = 3000
        h_pt = 4000

        waypoints = [[e_pt, n_pt, h_pt],
                    [e_pt + 2000, n_pt + 5000, h_pt - 100],
                    [e_pt - 2000, n_pt + 15000, h_pt - 250],
                    [e_pt - 500, n_pt + 25000, h_pt]]

        ap = WaypointAutopilot(waypoints, stdout=True)

        step = 1/30
        extended_states = True
        u_seq = np.zeros((int(tmax / step), 4))
        res = run_f16_sim(init, tmax, ap, u_seq, step=step, extended_states=extended_states, integrator_str='rk45')

        return res["times"], res["states"], np.array(res['u_list'])

    """

    # state is [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    # u is: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    times, states, controls = py"simulate"()
    
    # only keep the first 13 columns of states (the others are integration variables)
    states = states[:, 1:13]

    # only keep the first 4 columns of controls
    controls = controls[:, 1:4]

    return times, states, controls
end

function deg2rad(x)
    return π / 180 * x
end

function rad2deg(x)
    return 180 / π * x
end

"""
    scale_bounds(scaler, safe_bounds, start_index, end_index)

Scale the bounds using the given scaler.

# Arguments
- `scaler`: A `UnitRangeTransform` object representing the scaler.
- `safe_bounds`: A matrix of safe bounds.
- `start_index`: The starting index.
- `end_index`: The ending index.

# Returns
A scaled matrix of bounds.

"""
function scale_bounds(scaler::UnitRangeTransform{Float64,Vector{Float64}}, safe_bounds::Matrix{Float64}, start_index::Int, end_index::Int)
    lower_bounds = reshape(Vector{Float64}([safe_bounds[li, 1] for li in start_index:end_index]), 1, :)
    upper_bounds = reshape(Vector{Float64}([safe_bounds[ui, 2] for ui in start_index:end_index]), 1, :)

    # Set NaN values to large number
    lower_bounds[isinf.(lower_bounds)] .= -1e100
    upper_bounds[isinf.(upper_bounds)] .= 1e100

    lower_bounds = StatsBase.transform(scaler, reshape(lower_bounds, :, 1))
    upper_bounds = StatsBase.transform(scaler, reshape(upper_bounds, :, 1))

    # Replace large values with Inf
    upper_bounds[upper_bounds.>1e50] .= Inf
    lower_bounds[lower_bounds.<-1e50] .= -Inf

    return reshape(lower_bounds, :, 1), reshape(upper_bounds, :, 1)
end

"""
    estimate_linear_system(Z::Matrix{Float64}, n::Int64)

Estimates the linear system parameters A and B using the given data matrix Z and the number of inputs n.

# Arguments
- `Z::Matrix{Float64}`: The data matrix containing the input-output data.
- `n::Int64`: The number of inputs in the linear system.

# Returns
- `A::Matrix{Float64}`: The estimated system matrix A.
- `B::Matrix{Float64}`: The estimated input matrix B.
"""
function estimate_linear_system(Z::Matrix{Float64}, n::Int64)
    p = size(Z, 2) - 1
    X_prime = Z[1:n, 2:end]

    # Solve the linear system using pseudo-inverse
    Theta = X_prime * pinv(Z[:, 1:p])

    A = Theta[:, 1:n]
    B = Theta[:, n+1:end]

    return A, B
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
    safe_bounds = [
        300 2500; # vt ft/s
        deg2rad(-10) deg2rad(45); # alpha
        deg2rad(-30) deg2rad(30); # beta 
        deg2rad(-90) deg2rad(90); # phi
        deg2rad(-30) deg2rad(30); # theta
        deg2rad(-180) deg2rad(180); # psi
        deg2rad(-180) deg2rad(180); # P
        deg2rad(-180) deg2rad(180); # Q
        deg2rad(-180) deg2rad(180); # R
        -Inf Inf; # pn ft
        -Inf Inf; # pe ft
        1000 50000; # h ft
        0 100; # pow
        0 1; # throt
        deg2rad(-10) deg2rad(10); # ele
        deg2rad(-15) deg2rad(15); # ail
        deg2rad(-10) deg2rad(10) # rud
    ] 


    # 1. run F16 waypoint simulation to collect data set 
    times, states, controls = run_f16_waypoint_sim()
    n, m = size(states, 2), size(controls, 2)
    t = length(times)
    Δt = times[2] - times[1]
    @show Δt
    t_horizon = round(Int64, 25 / Δt)
    @show t_horizon


    # 2. scale the data
    Z_unscaled = hcat(states, controls)' # Z is shaped as (n+m,t) where n is the number of states and m is the number of controls
    scaler = fit(UnitRangeTransform, Z_unscaled, dims=2)
    Z = StatsBase.transform(scaler, Z_unscaled)

    # scale the bounds as well
    lower_bounds, upper_bounds = scale_bounds(scaler, safe_bounds, 1, n + m)

    new_safe_bounds = zeros(size(safe_bounds))
    for i in 1:size(safe_bounds, 1)
        new_safe_bounds[i, 1] = lower_bounds[i]
        new_safe_bounds[i, 2] = upper_bounds[i]
    end
    safe_bounds = new_safe_bounds

    # 3. estimate the linear system
    A_hat, B_hat = estimate_linear_system(Z, n)

    # 4. create the InputOptimizationProblem
    problem = InputOptimizationProblem(Z, scaler, times, A_hat, B_hat, n, m, t, t_horizon, Δt, safe_bounds, ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow", "throt", "ele", "ail", "rud"])

    return problem
end