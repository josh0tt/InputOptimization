
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
    safe_bounds_unscaled = [
        500 600; # vt ft/s
        -10 20; # alpha
        -30 30; # beta
        -30 30; # phi
        -10 10; # theta
        -180 180; # psi
        -40 40; # P
        -40 40; # Q
        -40 40; # R
        -Inf Inf; # pn ft
        -Inf Inf; # pe ft
        3000 5000; # h ft
        0 50; # pow
        0 1; # throt
        -2.75 -1.75; # ele
        -1 1; # ail
        -1 1 # rud
    ]

    # 1. run F16 waypoint simulation to collect data set 
    # we don't need it this smooth, we should just limit how far from the starting init we can go
    delta_max = 0.05 # maximum change in control inputs between time steps
    times, states, controls = run_f16_waypoint_sim()
    n, m = size(states, 2), size(controls, 2)
    n_t = length(times)
    Δt = times[2] - times[1]
    t_horizon = round(Int64, 25 / Δt)

    # 2. scale the data
    Z_unscaled = hcat(states, controls)' # Z is shaped as (n+m,t) where n is the number of states and m is the number of controls
    scaler = fit(UnitRangeTransform, Z_unscaled, dims=2)
    Z = StatsBase.transform(scaler, Z_unscaled)

    # scale the bounds as well
    lower_bounds, upper_bounds = scale_bounds(scaler, safe_bounds_unscaled, 1, n + m)

    safe_bounds = zeros(size(safe_bounds_unscaled))
    for i in 1:size(safe_bounds, 1)
        safe_bounds[i, 1] = lower_bounds[i]
        safe_bounds[i, 2] = upper_bounds[i]
    end

    # 3. estimate the linear system
    A_hat, B_hat = estimate_linear_system(Z, n)

    # 4. create the InputOptimizationProblem
    problem = InputOptimizationProblem(Z, scaler, times, A_hat, B_hat, n, m, n_t, t_horizon, Δt, safe_bounds, safe_bounds_unscaled, delta_max, ["vt", "alpha", "beta", "phi", "theta", "psi", "P", "Q", "R", "pn", "pe", "h", "pow", "throt", "ele", "ail", "rud"])

    return problem
end