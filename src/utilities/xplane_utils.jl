using JuMP, MosekTools, LinearAlgebra, StatsBase, Gurobi, Interpolations, LaTeXStrings, Distributions

"""
    load_human_flown_data(safe_bounds::Matrix{Float64}, include_forces::Bool=true)

Load human flown data.

Parameters
----------
safe_bounds : Matrix{Float64}
    The safe bounds for the data.
include_forces : Bool, optional
    Whether to include forces in the loaded data. Default is `true`.

"""
function load_human_flown_data(safe_bounds::Matrix{Float64}, include_forces::Bool=true, data_path::String="data/short_cessna_data.npz")
    state_titles = ["Roll", "Pitch", "Yaw", "Roll Rate", "Pitch Rate", "Yaw Rate", "Vx Acf", "Vy Acf", "Vz Acf", "Alpha"]
    force_titles = ["Cx", "Cy", "Cz", "CL", "CM", "CN"]
    control_titles = ["Elevator", "Aileron", "Rudder", "Throttle"]

    flight_data = npzread(data_path) #npzread("../data/state_force_control_data_v1.npz")
    times = flight_data["times"]
    states = flight_data["states"]
    forces = flight_data["forces"]
    # combine states and forces to just states 
    if include_forces
        states = hcat(states, forces)
        state_titles = vcat(state_titles, force_titles)
    end
    controls = flight_data["controls"]
    # controls = controls[:, 1:end-1]  # Remove throttle
    titles = vcat(state_titles, control_titles)

    n = size(states, 2)
    m = size(controls, 2)
    t = size(states, 1)

    # states and forces are read in with time as the first Dimension
    # Z_t is shaped as (n+m,t) where n is the number of states and m is the number of controls
    Z_t = hcat(states, controls)'

    scaler = fit(UnitRangeTransform, Z_t, dims=2)
    Z_t = StatsBase.transform(scaler, Z_t)

    # Set bounds for states
    lower_bounds, upper_bounds = scale_bounds(scaler, safe_bounds, 1, n + m)

    new_safe_bounds = zeros(size(safe_bounds))
    for i in 1:size(safe_bounds, 1)
        new_safe_bounds[i, 1] = lower_bounds[i]
        new_safe_bounds[i, 2] = upper_bounds[i]
    end
    safe_bounds = new_safe_bounds

    return safe_bounds, times, Z_t, titles, scaler, n, m, t
end

function get_execution_times(new_times::LinRange{Float64,Int64}, t_horizon::Int64, start_time::Float64, t0::Float64)
    Δt = new_times[2] - new_times[1]
    current_t = time() - start_time + t0

    return [current_t + Δt * i for i in 1:t_horizon]
end

"""
    interpolate_data(times::Vector{Float64}, Z_t::Matrix{Float64}, t_horizon::Int64, start_time::Float64, t0::Float64, n::Int64, m::Int64)

Interpolates data using the given parameters.

# Arguments
- `times::Vector{Float64}`: A vector of time values.
- `Z_t::Matrix{Float64}`: Observed data.
- `t_horizon::Int64`: The time horizon.
- `start_time::Float64`: The start time.
- `t0::Float64`: The initial time.
- `n::Int64`: The state dimension.
- `m::Int64`: The control dimension.

# Returns
- An interpolated data matrix.

"""
function interpolate_data(times::Vector{Float64}, Z_t::Matrix{Float64}, t_horizon::Int64, start_time::Float64, t0::Float64, n::Int64, m::Int64)
    # Interpolate data to have uniform time steps
    new_times = LinRange(times[1], times[end], length(times))
    new_Z_t = zeros(size(Z_t))

    for i in 1:n+m
        interp = linear_interpolation(times, Z_t[i, :])
        new_Z_t[i, :] = interp(new_times)
    end

    execution_times = get_execution_times(new_times, t_horizon, start_time, t0)

    return new_Z_t, new_times, execution_times
end

"""
    estimate_linear_system(times::Vector{Float64}, Z_t::Matrix{Float64}, t_horizon::Int64, start_time::Float64, t0::Float64, n::Int64, m::Int64)

Estimates a linear system based on the given inputs.

# Arguments
- `times::Vector{Float64}`: A vector of time values.
- `Z_t::Matrix{Float64}`: Observed data.
- `t_horizon::Int64`: The time horizon.
- `start_time::Float64`: The start time.
- `t0::Float64`: The initial time.
- `n::Int64`: The state dimension.
- `m::Int64`: The control dimension.

# Returns
- The estimated linear system.

"""
function estimate_linear_system(times::Vector{Float64}, Z_t::Matrix{Float64}, t_horizon::Int64, start_time::Float64, t0::Float64, n::Int64, m::Int64)
    Z_t, new_times, execution_times = interpolate_data(times, Z_t, t_horizon, start_time, t0, n, m)

    p = size(Z_t, 2) - 1
    X_prime = Z_t[1:n, 2:end]

    # Solve the linear system using pseudo-inverse
    Theta = X_prime * pinv(Z_t[:, 1:p])

    A = Theta[:, 1:n]
    B = Theta[:, n+1:end]

    return A, B, Z_t, new_times, execution_times
end

# """
#     build_model(safe_bounds::Matrix{Float64}, Z_t::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int64, m::Int64, t::Int64, T::Int64, Z_cur::Matrix{Float64}, Sigma_0_inv::Matrix{Float64}, scaler::UnitRangeTransform{Float64, Vector{Float64}}, delta_max::Float64, method::String)

# Builds a model using the given parameters.

# # Arguments
# - `safe_bounds::Matrix{Float64}`: The safe bounds matrix.
# - `Z_t::Matrix{Float64}`: The Z_t matrix.
# - `A_hat::Matrix{Float64}`: The A_hat matrix.
# - `B_hat::Matrix{Float64}`: The B_hat matrix.
# - `n::Int64`: The number of states.
# - `m::Int64`: The number of controls.
# - `t::Int64`: The current time step.
# - `T::Int64`: The total number of time steps.
# - `Z_cur::Matrix{Float64}`: The current Z matrix.
# - `Sigma_0_inv::Matrix{Float64}`: The inverse of the initial covariance matrix.
# - `scaler::UnitRangeTransform{Float64, Vector{Float64}}`: The scaler object.
# - `delta_max::Float64`: The maximum value of control input changes.
# - `method::String`: The method used.

# # Returns
# - The built model.

# """
# function build_model(safe_bounds::Matrix{Float64}, Z_t::Matrix{Float64}, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, n::Int64, m::Int64, t::Int64, T::Int64, Z_cur::Matrix{Float64}, Sigma_0_inv::Matrix{Float64}, scaler::UnitRangeTransform{Float64,Vector{Float64}}, delta_max::Float64, method::String)
#     if method == "exact"
#         model = Model(Mosek.Optimizer)
#         @variable(model, Y[1:n+m, 1:n+m], PSD)
#     else
#         model = Model(Gurobi.Optimizer)
#     end

#     @variable(model, Z_ctrl[1:n+m, 1:T-t])

#     Z = [Z_t Z_ctrl]
#     Z_diff = Z - Z_cur
#     Z_diff_transpose = Z_diff'
#     W_hat = Z_cur * Z_cur' + Z_cur * Z_diff_transpose + Z_diff * Z_cur' + Sigma_0_inv

#     if method == "exact"
#         @objective(model, Min, tr(Y))
#         @constraint(model, [Y Diagonal(ones(n + m)); Diagonal(ones(n + m)) W_hat] >= 0, PSDCone())
#     else
#         @objective(model, Min, -tr(W_hat))
#     end

#     # Constraints
#     # Initial Z matching
#     @constraint(model, Z[:, 1:t] .== Z_t[:, 1:t])

#     # Dynamics and control constraints
#     for i in t:T-1
#         @constraint(model, Z[1:n, i+1] .== A_hat * Z[1:n, i] + B_hat * Z[n+1:end, i]) # Dynamics
#         @constraint(model, Z[n+1:end, i] - Z[n+1:end, i+1] .<= delta_max) # Control variation
#         @constraint(model, Z[n+1:end, i+1] - Z[n+1:end, i] .<= delta_max)
#     end

#     upper_bounds = safe_bounds[:, 2]
#     lower_bounds = safe_bounds[:, 1]
#     valid_bounds = .!isinf.(upper_bounds)
#     @constraint(model, lower_bounds[valid_bounds] .<= Z[valid_bounds, t+1:T])
#     @constraint(model, Z[valid_bounds, t+1:T] .<= upper_bounds[valid_bounds])


#     # add constraint that we want to end up with zero roll, pitch, and yaw
#     desired_end_state = zeros(n + m, 1)
#     # Set desired Vx to -65 
#     desired_end_state[9] = -65.0
#     desired_end_state = StatsBase.transform(scaler, desired_end_state)
#     println("Desired end state: ", desired_end_state)
#     @constraint(model, Z[1:3, T] .== desired_end_state[1:3])
#     # @constraint(model, Z[9, T] == desired_end_state[9])


#     return model
# end

# """
#     plan_control_inputs(safe_bounds, times, Z_t, scaler, start_time, t0, t_horizon, n, m, t, method="approx")

# Plan control inputs based on safe bounds, times, and other parameters.

# # Arguments
# - `safe_bounds::Matrix{Float64}`: The safe bounds for the control inputs.
# - `times::Vector{Float64}`: The times at which data was observed.
# - `Z_t::Matrix{Float64}`: Observed state and control data.
# - `scaler::UnitRangeTransform{Float64, Vector{Float64}}`: The scaler used to transform the state trajectories.
# - `start_time::Float64`: The start time.
# - `t0::Float64`: Initial observed data time.
# - `t_horizon::Int64`: The length of the planning horizon.
# - `n::Int64`: The number of states.
# - `m::Int64`: The number of control inputs.
# - `t::Int64`: The index of the current time point.
# - `method::String`: The method used for planning control inputs. Default is "approx".

# # Returns
# - `control_inputs::Matrix{Float64}`: The planned control inputs.

# """
# function plan_control_inputs(safe_bounds::Matrix{Float64}, times::Vector{Float64}, Z_t::Matrix{Float64}, scaler::UnitRangeTransform{Float64,Vector{Float64}}, start_time::Float64, t0::Float64, t_horizon::Int64, n::Int64, m::Int64, t::Int64, method::String="approx")
#     plan_time = time()

#     A_hat, B_hat, Z_t, new_times, execution_times = estimate_linear_system(times, Z_t, t_horizon, start_time, t0, n, m)

#     T = t + t_horizon

#     Sigma_0_inv = diagm(0 => ones(n + m)) # Assuming Sigma_0_inv is defined elsewhere
#     # Z_cur = [Z_t Z_t[:, end] .+ zeros(n+m, t_horizon)]
#     Z_cur = [Z_t Z_t[:, end] .+ rand(Normal(0, 1.0), n + m, t_horizon)]
#     max_iter = 10
#     delta_max = 0.01
#     tol = 1e-3
#     obj = Inf
#     values = Float64[]
#     W_hat = Nothing
#     Z = Nothing
#     Z_ctrl_val = Nothing

#     build_time = time()
#     model = build_model(safe_bounds, Z_t, A_hat, B_hat, n, m, t, T, Z_cur, Sigma_0_inv, scaler, delta_max, method)
#     build_end_time = time()
#     println("Time spent in build_model: ", build_end_time - build_time)

#     for iter in 1:max_iter
#         println("#"^30)
#         println("Iteration $iter/$max_iter")
#         println("#"^30)

#         # Solve the problem
#         JuMP.optimize!(model)

#         # Check solver status and update Z_cur if feasible
#         println("Solver status: ", termination_status(model))
#         if termination_status(model) == MOI.INFEASIBLE_OR_UNBOUNDED
#             println("Infeasible problem. Stabilizing aircraft...")
#             @warn("Infeasible problem. Stabilizing aircraft...")
#             stable_control_traj = stabilize_aircraft(safe_bounds, times, Z_t, scaler, start_time, t0, t_horizon, n, m, t, A_hat, B_hat, delta_max)
#             Z_ctrl_val = zeros(n + m, t_horizon)
#             Z_ctrl_val[n+1:end, :] .= stable_control_traj
#             unscaled_Z_ctrl = StatsBase.reconstruct(scaler, Z_ctrl_val)
#             control_traj = unscaled_Z_ctrl[n+1:end, :]'

#             return control_traj, zeros(n + m, T), execution_times, true
#         end

#         # Check solver status and update Z_cur if feasible
#         current_obj = objective_value(model)
#         push!(values, current_obj)

#         Z_ctrl_val = value.(model[:Z_ctrl])
#         Z_cur = [Z_t Z_ctrl_val]
#         Z = [Z_t model[:Z_ctrl]]

#         # Update W_hat for model objective 
#         Z_diff = Z - Z_cur
#         Z_diff_transpose = Z_diff'
#         W_hat = Z_cur * Z_cur' + Z_cur * Z_diff_transpose + Z_diff * Z_cur' + Sigma_0_inv
#         set_objective_function(model, -tr(W_hat))


#         # Convergence check
#         if abs(obj - current_obj) < tol
#             println("Converged after $iter iterations")
#             break
#         end
#         obj = current_obj

#     end

#     unscaled_Z_ctrl = StatsBase.reconstruct(scaler, Z_ctrl_val)
#     control_traj = unscaled_Z_ctrl[n+1:end, :]'

#     end_time = time()
#     println("Time spent in plan_control_inputs: ", end_time - plan_time)
#     return control_traj, Z_cur, execution_times, false
# end


function stabilize_aircraft(safe_bounds::Matrix{Float64}, times::Vector{Float64}, Z_t::Matrix{Float64}, scaler::UnitRangeTransform{Float64,Vector{Float64}}, start_time::Float64, t0::Float64, t_horizon::Int64, n::Int64, m::Int64, t::Int64, A_hat::Matrix{Float64}, B_hat::Matrix{Float64}, delta_max::Float64)
    # Using the current A_hat and B_hat, plan control inputs that drive the aircraft to the desired state
    roll_des = 0.0
    pitch_des = 0.0
    yaw_des = 0.0
    roll_rate_des = 0.0
    pitch_rate_des = 0.0
    yaw_rate_des = 0.0

    desired_end_state = zeros(n + m, 1)
    desired_end_state[1:6] = [roll_des, pitch_des, yaw_des, roll_rate_des, pitch_rate_des, yaw_rate_des]
    desired_end_state = StatsBase.transform(scaler, desired_end_state)

    # Plan control inputs to drive the aircraft to the desired state
    model = Model(Gurobi.Optimizer)

    # Decision variables: control inputs over the horizon
    @variable(model, u[1:m, 1:t_horizon])

    # State variables over the horizon
    @variable(model, x[1:n, 1:t_horizon+1])

    # Initial state conditions
    @constraint(model, x[:, 1] .== Z_t[1:n, t])
    @constraint(model, u[:, 1] .== Z_t[n+1:end, t])

    # Dynamics constraints over the prediction horizon
    for k in 1:t_horizon
        @constraint(model, x[:, k+1] .== A_hat * x[:, k] + B_hat * u[:, k])
    end

    for i in 1:t_horizon-1
        @constraint(model, u[:, i] - u[:, i+1] .<= delta_max) # Control variation
        @constraint(model, u[:, i+1] - u[:, i] .<= delta_max)
    end


    # Objective: minimize the deviation from the desired end state (first 6 states)
    @objective(model, Min, sum((x[1:6, t_horizon+1] .- desired_end_state[1:6]) .^ 2))

    # Solve the optimization problem
    JuMP.optimize!(model)

    # Check if the model was solved
    if termination_status(model) == MOI.OPTIMAL
        # Extract the optimal control inputs
        optimal_u = value.(u)
        return optimal_u  # Return the first set of control inputs
    else
        error("The optimization problem did not solve successfully.")
    end

end


"""
    initialize_plots(safe_bounds::Matrix{Float64}, times::Vector{Float64}, Z_t::Matrix{Float64}, titles::Vector{String}, n::Int64, m::Int64, include_forces::Bool)

Initialize and display a plot with subplots for visualizing data.

# Arguments
- `safe_bounds::Matrix{Float64}`: A matrix containing the lower and upper bounds.
- `times::Vector{Float64}`: The times at which data was observed.
- `Z_t::Matrix{Float64}`: Observed state and control data.
- `titles::Vector{String}`: A vector of titles.
- `n::Int64`: The number of states.
- `m::Int64`: The number of control inputs.
- `include_forces::Bool`: A flag indicating whether to include forces in the plot.

# Returns
- `p`: The plot object.
- `plts`: An array of trace objects.
- `layout`: The layout object.
- `plt_t0`: The length of the `times` vector.
- `n_traces`: The number of traces.
- `obj_hist`: An array of objective values.

"""
function initialize_plots(safe_bounds::Matrix{Float64}, times::Vector{Float64}, Z_t::Matrix{Float64}, titles::Vector{String}, n::Int64, m::Int64, include_forces::Bool)
    plt_t0 = length(times)
    n_traces = n + m

    # Determine the number of rows and columns for subplots based on the number of traces
    if include_forces
        n_rows = ceil(Int, sqrt(n_traces))  # or any other way to decide rows and columns
        n_cols = ceil(Int, n_traces / n_rows)
    else
        n_rows = 5 #ceil(Int, n_traces / 2)  
        n_cols = 3 #ceil(Int, n_traces / n_rows)
    end

    # Create a subplot layout
    layout = Layout(; grid=attr(rows=n_rows, columns=n_cols, pattern="independent"), showlegend=false)

    # Initialize an empty vector for traces
    plts = GenericTrace{Dict{Symbol,Any}}[]

    for i in 1:n_traces
        trace_predicted = PlotlyJS.scatter(x=[], y=[], mode="markers", name="Predicted $(titles[i])", xaxis="x$i", yaxis="y$i", line=attr(color="#fca503"), showlegend=false) # #ff82f9
        push!(plts, trace_predicted)
    end

    for i in 1:n_traces
        trace_actual = PlotlyJS.scatter(x=times[1:plt_t0], y=Z_t[i, 1:plt_t0], mode="lines", name="Actual $(titles[i])", xaxis="x$i", yaxis="y$i", line=attr(color="#009af9"), showlegend=false)
        push!(plts, trace_actual)
    end

    # plot horizontal line safety bounds as black dashed lines
    for i in 1:n_traces
        trace_lower = PlotlyJS.scatter(x=[times[1], times[end]], y=[safe_bounds[i, 1], safe_bounds[i, 1]], mode="lines", name="Lower Bound", xaxis="x$i", yaxis="y$i", line=attr(color="#000000", dash="dash"), showlegend=false)
        push!(plts, trace_lower)
    end

    for i in 1:n_traces
        trace_upper = PlotlyJS.scatter(x=[times[1], times[end]], y=[safe_bounds[i, 2], safe_bounds[i, 2]], mode="lines", name="Upper Bound", xaxis="x$i", yaxis="y$i", line=attr(color="#000000", dash="dash"), showlegend=false)
        push!(plts, trace_upper)
    end

    obj_hist = compute_obj_hist(n, m, plt_t0, Z_t)
    trace_obj = PlotlyJS.scatter(x=times[1:end-1], y=obj_hist, mode="lines", name="Objective", xaxis="x$(n_traces+1)", yaxis="y$(n_traces+1)", line=attr(color="#000000"), showlegend=false)
    push!(plts, trace_obj)

    titles = vcat(titles, "Objective")

    layout = Layout()

    for i in 1:n_traces+1
        axis_idx = string(i)
        layout["xaxis$axis_idx"] = attr(domain=[((i - 1) % n_cols) / n_cols, (i % n_cols) / n_cols - 0.02], anchor="y$axis_idx")
        layout["yaxis$axis_idx"] = attr(domain=[1 - ceil(i / n_cols) / n_rows, 1 - (ceil(i / n_cols) - 1) / n_rows - 0.1], anchor="x$axis_idx")
    end

    annotations = []

    # Loop over each subplot to create annotations
    for i in 1:n_traces+1
        row = floor((i - 1) / n_cols)

        # Adjusted y position calculation to bring the title closer to the subplot
        # Here, we're making the offset a bit smaller to reduce the gap.
        if include_forces
            y_position = 1 - (row / n_rows) - (1 / n_rows * 0.5)  # Adjust the multiplier to control the gap
        else
            y_position = 1 - (row / n_rows) - (1 / n_rows * 0.5)  # Adjust the multiplier to control the gap
        end

        push!(annotations, attr(
            xref="x$(i) domain",
            yref="paper",
            x=0.5,  # Centered within the subplot's x-axis domain
            y=y_position,
            text=titles[i],  # Subplot title
            showarrow=false,
            xanchor="center",
            yanchor="bottom",
            font=attr(size=20)
        ))
    end

    layout[:annotations] = annotations


    p = PlotlyJS.plot(plts, layout)
    display(p)

    return p, plts, layout, plt_t0, n_traces, obj_hist
end

"""
    compute_obj_hist(n::Int64, m::Int64, t::Int64, Z_t::Matrix{Float64})

Compute the objective history.

# Arguments
- `n::Int64`: The number of states.
- `m::Int64`: The number of control inputs.
- `t::Int64`: The number of time steps.
- `Z_t::Matrix{Float64}`: The matrix of observed data.

# Returns
- `obj_hist::Vector{Float64}`: The objective history as a vector of type `Vector{Float64}`.

"""
function compute_obj_hist(n::Int64, m::Int64, t::Int64, Z_t::Matrix{Float64})
    st = 25
    obj_hist = zeros(t - st)
    residual_norm_hist = []
    for i in st:t-1
        sigma, residual_norm = compute_sigma(Z_t[:, 1:i], n)
        obj = tr(sigma^(2) * inv(Z_t[:, 1:i-1] * Z_t[:, 1:i-1]'))
        # obj = log(det(sigma^(2) * inv(Z_t[:, 1:i-1] * Z_t[:, 1:i-1]')))
        obj_hist[i-st+1] = obj
        push!(residual_norm_hist, residual_norm)
    end

    return log.(obj_hist)#, residual_norm_hist
end