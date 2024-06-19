using PyCall
using DataFrames
using LinearAlgebra
using Statistics
using DelimitedFiles
using NPZ
using JLD2
using PlotlyJS

struct FlightData
    Z_k::Matrix{Float64}
    times::Vector{Float64}
end

struct PlannedData
    planned_traj::Matrix{Float64}
    planned_times::Vector{Float64}
end

mutable struct SharedData
    flight_data::FlightData
    planned_data::PlannedData
    t0::Float64
    start_time::Float64
end

function run_xplane()

    lk = ReentrantLock()

    function collect_data(shared_data::SharedData, scaler::UnitRangeTransform{Float64, Vector{Float64}}, include_forces::Bool)
        t0 = shared_data.t0
        start_time = shared_data.start_time

        Δt = mean(diff(shared_data.flight_data.times))
        # start lpf timer to collect data
        data_collection_time = Δt
        i = 0 
        while true
            loop_time = time()
            # Collect data asynchronously
            state = py"get_state(client)"
            if include_forces
                force = py"get_force(client)"
                force = reshape(force, :, 1)
                state = reshape(state, :, 1)
                state = vcat(state, force)
            else
                state = reshape(state, :, 1)
            end
            control = py"get_control(client)"
            time_ = time() - start_time + t0

            control = reshape(control, :, 1)
            
            # Combine into a single column
            combined_data = StatsBase.transform(scaler, vcat(state, control))

            lock(lk)
            try
                times = shared_data.flight_data.times
                Z_k = shared_data.flight_data.Z_k
                
                times = vcat(times, time_)      # Append to times
                Z_k = hcat(Z_k, combined_data)  # Append to Z_k
                shared_data.flight_data = FlightData(Z_k, times)
            finally
                unlock(lk)
            end

            data_collection_time = 0.1*(time() - loop_time) + 0.9*data_collection_time
            if i % 100 == 0
                println("Time to collect data: ", data_collection_time)
            end
            i += 1

            # Sleep or wait for the next data collection point if necessary
            sleep(max(Δt - (time() - loop_time), 0.0))
            # sleep(0.1)
        end
    end

    function plan_trajectory(shared_data::SharedData, problem::InputOptimizationProblem)
        safe_bounds, scaler, t_horizon, n, m, n_t = problem.safe_bounds, problem.scaler, problem.t_horizon, problem.n, problem.m, problem.n_t
        t0 = shared_data.t0
        start_time = shared_data.start_time
        data_copy = deepcopy(shared_data.flight_data)
        sleep(2)
        try
            while true
                loop_time = time()
                
                lock(lk)
                try
                    data_copy = deepcopy(shared_data.flight_data)
                finally
                    unlock(lk)
                end

                Z_k = data_copy.Z_k
                times = data_copy.times
                n_t = size(Z_k, 2)

                A_hat, B_hat, Z_k, _, execution_times = estimate_linear_system(times, Z_k, t_horizon, start_time, t0, n, m)
                𝒩 = InputOptimization.fit_process_noise(Z_k, A_hat, B_hat, n, m)
                problem = InputOptimizationProblem(problem.rng, Z_k, scaler, times, A_hat, B_hat, 𝒩, problem.n, problem.m, n_t, problem.t_horizon, mean(diff(execution_times)), problem.safe_bounds, problem.safe_bounds_unscaled, problem.delta_maxs, problem.max_As, problem.f_min, problem.f_max, problem.row_names, problem.equal_time_constraint)
                Z_cur, infeasible_flag = plan_control_inputs(problem, "approx", "xplane")
                if infeasible_flag
                    @warn("Infeasible problem. Stabilizing aircraft...")    
                    stable_control_traj = stabilize_aircraft(problem.safe_bounds, problem.times, Z_k, problem.scaler, start_time, t0, problem.t_horizon, problem.n, problem.m, problem.t, problem.A_hat, problem.B_hat, problem.delta_maxs)
                    control_traj = StatsBase.reconstruct(scaler, vcat(zeros(problem.n, problem.t_horizon), stable_control_traj))[problem.n+1:end, :]'
                else
                    control_traj = StatsBase.reconstruct(scaler, Z_cur[:, problem.n_t+1:end])[problem.n+1:end, :]'
                    lock(lk)
                    try
                        shared_data.planned_data = PlannedData(Z_cur[:, n_t:end], execution_times)
                    finally
                        unlock(lk)
                    end
                end

                println("Time to replan: ", time() - loop_time)

                println("Difference in time:", time() - start_time + t0 - times[end])

                for i in 1:size(control_traj, 1)
                    println("Sending control input: $(i) / $(size(control_traj, 1))")
                    execute_time = execution_times[i]

                    elev = control_traj[i, 1]
                    ail = control_traj[i, 2]
                    rud = control_traj[i, 3]
                    throttle = control_traj[i, 4] # -998

                    # Wait until the execution time
                    current_t = time() - start_time + t0
                    while current_t < execute_time
                        # println("Sleeping for ", execute_time - current_t)
                        sleep(execute_time - current_t)
                        current_t = time() - start_time + t0
                    end

                    py"client.sendCTRL"([elev, ail, rud, throttle, -998, -998, -998])

                end

                sleep(0.01)
            end
        catch e
            println("Error: ", e)
            println("Line number: ", catch_backtrace())
        end
    end

    function plot_data(shared_data::SharedData, safe_bounds::Matrix{Float64}, p, plts, layout, plt_t0::Int64, n_traces, obj_hist, n::Int64, m::Int64)
        h = 0
        plot_data_copy = deepcopy(shared_data)
        plotting_time = 0.0
        i = 0
        plt_t0 = 1
        try
            while true
                plot_time = time()

                lock(lk)
                try
                    plot_data_copy = deepcopy(shared_data)
                finally
                    unlock(lk)
                end

                real_trajs = plot_data_copy.flight_data.Z_k
                real_times = plot_data_copy.flight_data.times

                planned_traj = plot_data_copy.planned_data.planned_traj
                planned_times = plot_data_copy.planned_data.planned_times

                n_t = size(real_trajs, 2)
                if n_t > 500
                    # save the data to a file
                    println("Saving data to file...")
                    data_path = joinpath(@__DIR__, "..", "data", "xplane", "xplane_data.jld2")
                    save(data_path, "Z_k", real_trajs, "times", real_times)
                    println("Data saved to file.")
                    break
                end

                # check if planned trajectory is empty
                if !isempty(planned_times)

                    # # Updating planned trajectory every iteration using dynamics and current state 
                    # # but control inputs remain the same from previous planning iteration 
                    # x0 = real_trajs[:, end]
                    # x = zeros(size(planned_traj))
                    # x[:, 1] = x0
                    # for i in 1:size(planned_traj, 2)-1
                    #     println("Updating planned trajectory for time: ", i)
                    #     x[1:problem.n, i+1] = problem.A_hat * x[1:problem.n, i] + problem.B_hat * planned_traj[problem.n+1:end, i]
                    # end
                    # x[problem.n+1:end, :] = planned_traj[problem.n+1:end, :]
                    # planned_traj = x
                    # planned_times = real_times[end] .+ problem.Δt * collect(1:size(planned_traj, 2))
                    # println("Planned trajectory updated.")

                    for i in 1:n_traces
                        plts[i][:x] = planned_times
                        plts[i][:y] = planned_traj[i, :]
                    end
                    end_time = max(real_times[end], planned_times[end]) + 10
                else
                    end_time = real_times[end] + 10
                end

                for i in 1:n_traces
                    plts[i+n_traces][:x] = real_times[plt_t0-h:n_t]
                    plts[i+n_traces][:y] = real_trajs[i, plt_t0-h:n_t]
                end

                for i in 1:n_traces
                    plts[i+2*n_traces][:x] = [real_times[plt_t0-h], end_time]
                    plts[i+2*n_traces][:y] = [safe_bounds[i, 1], safe_bounds[i, 1]]
                end

                for i in 1:n_traces
                    plts[i+3*n_traces][:x] = [real_times[plt_t0-h], end_time]
                    plts[i+3*n_traces][:y] = [safe_bounds[i, 2], safe_bounds[i, 2]]
                end

                obj_hist = compute_obj_hist(n, m, n_t, real_trajs[:, 1:n_t])
                plts[4*n_traces+1][:x] = real_times[1:n_t]
                plts[4*n_traces+1][:y] = obj_hist

                # Use `react!` to update the plot with new data
                react!(p, plts, layout)

                plotting_time = 0.1*(time() - plot_time) + 0.9*plotting_time
                if i % 100 == 0
                    println("Time to plot: ", plotting_time)
                end
                i += 1
                
                sleep(0.1)
            end
        catch e
            println("Error: ", e)
        end
    end


    println("Using $(Threads.nthreads()) threads")
    include_forces = false

    t_horizon = 25
    if include_forces
        safe_bounds_unscaled = [-30 30;       # Roll 1
                    -25 25;                   # Pitch 2
                    -10 10;                   # Yaw 3 
                    -10 10;                   # Roll Rate 4 
                    -10 10;                   # Pitch Rate 5
                    -10 10;                   # Yaw Rate 6
                    -Inf Inf;                 # Vx Acf 7
                    -Inf Inf;                 # Vy Acf 8
                    -80 -40;                  # Vz Acf 9
                    -5 10;                    # Alpha 10
                    -Inf Inf;                 # Cx 11
                    -Inf Inf;                 # Cy 12
                    -Inf Inf;                 # Cz 13
                    -Inf Inf;                 # CL 14
                    -Inf Inf;                 # CM 15
                    -Inf Inf;                 # CN 16
                    -1 1;                     # Elevator 17
                    -1 1;                     # Aileron 18
                    -1 1;                     # Rudder 19
                     0 1]                     # Throttle 20
    else
        safe_bounds_unscaled = [-30 30;       # Roll 1
                    -25 25;                   # Pitch 2
                    -10 10;                   # Yaw 3
                    -10 10;                   # Roll Rate 4
                    -10 10;                   # Pitch Rate 5
                    -10 10;                   # Yaw Rate 6
                    -Inf Inf;                 # Vx Acf 7
                    -Inf Inf;                 # Vy Acf 8
                    -80 -40;                  # Vz Acf 9
                    -5 10;                    # Alpha 10
                    -1 1;                     # Elevator 11
                    -1 1;                     # Aileron 12
                    -1 1;                     # Rudder 13
                     0 1]                     # Throttle 14
    end

    #################################################################
    # Python Setup
    #################################################################
    pushfirst!(PyVector(pyimport("sys")."path"), "")

    py"""
    import sys
    sys.path.append("/../xplane")
    """

    py"""
    from xplane.xpc3 import *
    import time
    import pickle 
    import numpy as np
    import math
    import itertools
    import socket
    import pandas as pd
    import pyautogui
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    def get_state(client):
        roll = client.getDREF("sim/flightmodel/position/phi")[0]
        pitch = client.getDREF("sim/flightmodel/position/theta")[0]
        yaw = client.getDREF("sim/flightmodel/position/beta")[0] # beta is the heading relative to the flown path (yaw) wheras psi is true heading of the aircraft in degrees from the Z axis #client.getDREF("sim/flightmodel/position/psi")[0]
        roll_dot = client.getDREF("sim/flightmodel/position/P")[0]
        pitch_dot = client.getDREF("sim/flightmodel/position/Q")[0]
        yaw_dot = client.getDREF("sim/flightmodel/position/R")[0]
        vx = client.getDREF("sim/flightmodel/forces/vx_acf_axis")[0]
        vy = client.getDREF("sim/flightmodel/forces/vy_acf_axis")[0]
        vz = client.getDREF("sim/flightmodel/forces/vz_acf_axis")[0]
        alpha = client.getDREF("sim/flightmodel/position/alpha")[0]

        return [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot, vx, vy, vz, alpha]

    def get_force(client):
        Fx = client.getDREF("sim/flightmodel/forces/fside_aero")[0]
        Fy = client.getDREF("sim/flightmodel/forces/fnrml_aero")[0]
        Fz = client.getDREF("sim/flightmodel/forces/faxil_aero")[0]
        L = client.getDREF("sim/flightmodel/forces/L_aero")[0]
        M = client.getDREF("sim/flightmodel/forces/M_aero")[0]
        N = client.getDREF("sim/flightmodel/forces/N_aero")[0]
        
        # compute dynamic pressure in units of N/(m*m). (dynamic pressure at current flight condition)
        rho = client.getDREF("sim/weather/rho")[0] # density of the air in kg/cubic meters.
        vx = client.getDREF("sim/flightmodel/forces/vx_acf_axis")[0]
        vy = client.getDREF("sim/flightmodel/forces/vy_acf_axis")[0]
        vz = client.getDREF("sim/flightmodel/forces/vz_acf_axis")[0]
        Q = 0.5 * rho * (vx**2 + vy**2 + vz**2)

        # the following two values are for Cessna 172 and are output from X-plane by going to Developer --> Aircraft Performance --> Find Pitch Stability Derivative
        # NOTE: these are only required to report the dimensionless coefficients since X-plane doesn't return them automatically.
        Sref = 18.8164      # m^2 (total wing area of the craft, based on all flying surfaces with a dihedral of less than 45 degrees, meaning wings and horizontal stabs but not vertical stabs)
        Cref = 1.13         # m (average mean aerodynamic chord of the craft, based on all flying surfaces with a dihedral of less than 45 degrees, meaning wings and horizontal stabs but not vertical stabs)

        # compute the coefficients
        Cx = Fx / (Q * Sref)
        Cy = Fy / (Q * Sref)
        Cz = Fz / (Q * Sref)
        Cl = L / (Q * Sref * Cref)
        Cm = M / (Q * Sref * Cref)
        Cn = N / (Q * Sref * Cref)

        return [Cx, Cy, Cz, Cl, Cm, Cn]

    def get_control(client):
        ctrl = client.getCTRL()
        current_elev = ctrl[0]
        current_ail = ctrl[1]
        current_rud = ctrl[2]
        current_throttle = ctrl[3]

        return [current_elev, current_ail, current_rud, current_throttle]
        # return [current_elev, current_ail, current_rud]

    def send_ctrl_input(client, elev, ail, rud, throttle):    
        client.sendCTRL([elev, ail, rud, throttle, -998, -998, -998])                

    def load_saved_flight(client):
        print("Switching to X-Plane in 1 second...")
        time.sleep(1)  # Wait for a few seconds to switch to X-Plane

        # check if position matches that of stored flight
        pos = client.getPOSI()
        # check if pos[2] is approximately 2393.251909600571
        while not np.isclose(pos[2], 2518.7004485120997, atol=1e-2):
            print("Current Lat, Lon, Alt: ", pos[0], pos[1], pos[2])
            # # Simulate Shift+Z key press
            # print("Pressing Shift+z")
            # pyautogui.hotkey('shift', 'z')

            # Simulate Shift+b key press
            print("Pressing Shift+b")
            pyautogui.hotkey('shift', 'b')
            time.sleep(1)
            pos = client.getPOSI()

    client = XPlaneConnect()

    """
    #################################################################
    # Run Xplane
    #################################################################

    # Load data
    safe_bounds, times, Z_k, titles, scaler, n, m, n_t = load_human_flown_data(safe_bounds_unscaled, include_forces)
    
    p, plts, layout, plt_t0, n_traces, obj_hist = initialize_plots(safe_bounds, times, Z_k, titles, n, m, include_forces)

    t0 = times[end]
    start_time = time()

    # call plan_control_inputs once here to precompile the function
    A_hat, B_hat, Z_k, _, execution_times = estimate_linear_system(times, Z_k, t_horizon, start_time, t0, n, m)
    max_As = find_max_As(n, m, StatsBase.reconstruct(scaler, Z_k), safe_bounds_unscaled)
    delta_maxs = 0.01 .* ones(m)
    𝒩 = InputOptimization.fit_process_noise(Z_k, A_hat, B_hat, n, m)
    problem = InputOptimizationProblem(MersenneTwister(12345), Z_k, scaler, times, A_hat, B_hat, 𝒩, n, m, n_t, t_horizon, mean(diff(times)), safe_bounds, safe_bounds_unscaled, delta_maxs, max_As, 0.1, 1.7, titles, false)   
    _, _ = plan_control_inputs(problem, "approx", "xplane")

    # save problem to file
    problem_path = joinpath(@__DIR__, "..", "data", "xplane", "xplane_problem.jld2")
    save(problem_path, "problem", problem)

    shared_data = SharedData(FlightData(Z_k, times), PlannedData(Matrix{Float64}(undef,0,0), Vector{Float64}()), t0, start_time)
    
    # Unpause X-Plane to start the episode
    py"client.pauseSim(False)"
    py"client.pauseSim(True)"
    py"load_saved_flight(client)"
    py"client.pauseSim(False)"

    # Start data collection and trajectory replanning
    data_task = Threads.@spawn collect_data(shared_data, scaler, include_forces)
    replan_task = Threads.@spawn plan_trajectory(shared_data, problem)
    plot_task = Threads.@spawn plot_data(shared_data, safe_bounds, p, plts, layout, plt_t0, n_traces, obj_hist, n, m)

    wait(data_task)
    wait(replan_task)
    wait(plot_task)
end