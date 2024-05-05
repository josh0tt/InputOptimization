function run_f16_waypoint_sim(step=1/15, gif=false)
    pushfirst!(PyVector(pyimport("sys")."path"), "")

    py"""
    import math
    import numpy as np
    from numpy import deg2rad
    import matplotlib.pyplot as plt

    from aerobench.run_f16_sim import run_f16_sim
    from aerobench.visualize import anim3d, plot
    from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot

    def simulate(step, gif):
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
        psi = 0           # Yaw angle from North (rad)

        # Build Initial Condition Vectors
        # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
        init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]

        # make waypoint list
        # e_pt = 1000
        # n_pt = 3000
        # h_pt = 4000

        # tmax = 200 # simulation time
        # waypoints = [[e_pt, n_pt, h_pt],
        #             [e_pt + 2000, n_pt + 5000, h_pt - 100],
        #             [e_pt - 2000, n_pt + 15000, h_pt - 250],
        #             [e_pt - 500, n_pt + 25000, h_pt]]
        tmax = 300 # simulation time
        # waypoints = [[e_pt, n_pt, h_pt],
        #             [e_pt + 2000, n_pt + 5000, h_pt - 100],
        #             [e_pt, n_pt + 25000, h_pt - 2500],
        #             [e_pt - 25000, n_pt + 30000, h_pt + 200],
        #             [e_pt - 20000, n_pt + 35000, h_pt]]
        e_pt = 0
        n_pt = 0
        h_pt = alt
        # waypoints = [[e_pt, n_pt + 5000, h_pt + 1000],
        #         [e_pt, n_pt + 10000, h_pt],
        #         [e_pt, n_pt + 15000, h_pt - 1000],
        #         [e_pt, n_pt + 20000, h_pt],
        #         [e_pt + 10000, n_pt + 20000, h_pt],
        #         [e_pt + 20000, n_pt + 20000, h_pt],
        #         [e_pt + 35000, n_pt + 20000, h_pt+1000]]
        waypoints = [[e_pt, n_pt + 5000, h_pt + 1000],
                     [e_pt, n_pt + 10000, h_pt],
                     [e_pt, n_pt + 15000, h_pt - 1000],
                     [e_pt, n_pt + 20000, h_pt],
                     [e_pt, n_pt + 25000, h_pt + 1000],
                     [e_pt, n_pt + 30000, h_pt],
                     [e_pt, n_pt + 35000, h_pt - 1000],
                     [e_pt, n_pt + 40000, h_pt],
                     [e_pt + 10000, n_pt + 40000, h_pt],
                     [e_pt + 20000, n_pt + 40000, h_pt],
                     [e_pt + 30000, n_pt + 40000, h_pt+1000],
                     [e_pt + 40000, n_pt + 40000, h_pt],
                     [e_pt + 50000, n_pt + 40000, h_pt-1000],
                     [e_pt + 70000, n_pt + 40000, h_pt]]



        ap = WaypointAutopilot(waypoints, stdout=True)

        extended_states = True
        u_seq = np.zeros((int(tmax / step)+1, 4))
        print("size of u_seq: ", u_seq.shape)
        res = run_f16_sim(init, tmax, ap, u_seq, step=step, extended_states=extended_states, integrator_str='rk45')

        ###########################
        # gif
        ###########################
        filename = 'waypoint.gif'

        if filename.endswith('.mp4'):
            skip_override = 4
        elif filename.endswith('.gif'):
            skip_override = 8 #15
        else:
            skip_override = 30
    
        anim_lines = []
        modes = res['modes']
        modes = modes[0::skip_override]
    
        def init_extra(ax):
            'initialize plot extra shapes'
    
            l1, = ax.plot([], [], [], 'bo', ms=8, lw=0, zorder=50)
            anim_lines.append(l1)
    
            l2, = ax.plot([], [], [], 'lime', marker='o', ms=8, lw=0, zorder=50)
            anim_lines.append(l2)
    
            return anim_lines
    
        def update_extra(frame):
            'update plot extra shapes'
    
            mode_names = ['Waypoint 1', 'Waypoint 2', 'Waypoint 3']
    
            done_xs = []
            done_ys = []
            done_zs = []
    
            blue_xs = []
            blue_ys = []
            blue_zs = []
    
            for i, mode_name in enumerate(mode_names):
                if modes[frame] == mode_name:
                    blue_xs.append(waypoints[i][0])
                    blue_ys.append(waypoints[i][1])
                    blue_zs.append(waypoints[i][2])
                    break
    
                done_xs.append(waypoints[i][0])
                done_ys.append(waypoints[i][1])
                done_zs.append(waypoints[i][2])
    
            anim_lines[0].set_data(blue_xs, blue_ys)
            anim_lines[0].set_3d_properties(blue_zs)
    
            anim_lines[1].set_data(done_xs, done_ys)
            anim_lines[1].set_3d_properties(done_zs)

        if gif:
            plot.plot_overhead(res, waypoints=waypoints)
            overhead_filename = 'waypoint_overhead.png'
            plt.savefig(overhead_filename)
            print(f"Made {overhead_filename}")
            plt.close()

            plot.plot_single(res, 'alt', title='Altitude (ft)')
            alt_filename = 'waypoint_altitude.png'
            plt.savefig(alt_filename)
            print(f"Made {alt_filename}")
            plt.close()

            anim3d.make_anim(res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
                            elev=27, azim=-107, skip_frames=skip_override,
                            chase=True, fixed_floor=True, init_extra=init_extra, update_extra=update_extra)

        return res["times"], res["states"], np.array(res['u_list']), np.array(res['Nz_list'])

    """

    # state is [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    # u is: throt, ele, ail, rud, Nz_ref, ps_ref, Ny_r_ref
    times, states, controls, Nz = py"simulate"(step, gif)
    
    # only keep the first 13 columns of states (the others are integration variables)
    states = states[:, 1:13]
    for i in 2:9
        states[:, i] = rad2deg(states[:, i])
    end
    # include Nz in states 
    # states = hcat(states, Nz) 

    # only keep the first 4 columns of controls
    controls = controls[:, 1:4]
    for i in 2:4
        controls[:, i] = rad2deg(controls[:, i])
    end

    return times, states, controls
end

function run_f16_sim(problem::InputOptimizationProblem, Z_planned::Matrix{Float64}, method_name::String="none", gif=false)
    n_t = problem.n_t
    Z_planned_unscaled = StatsBase.reconstruct(problem.scaler, Z_planned)
    # we go from n_t to end because the first value was the last control input from the collected data
    u_seq = Z_planned_unscaled[problem.n+1:end, n_t:end]'
    # u_seq = Z_planned_unscaled[problem.n+1:end, n_t+1:end]'

    # Initial Conditions is the last state from the collected data
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = Z_planned_unscaled[1:problem.n, n_t]
    # init is in rad for angles, but Z_planned_unscaled is in deg
    for i in 2:9
        init[i] = deg2rad(init[i])
    end
    alt = init[12]

    pushfirst!(PyVector(pyimport("sys")."path"), "")

    py"""
    import math
    import numpy as np
    from numpy import deg2rad

    from aerobench.run_f16_sim import run_f16_sim
    from aerobench.visualize import anim3d, plot
    from aerobench.examples.straight_and_level.run import StraightAndLevelAutopilot

    def simulate(init, alt, u_seq, tmax, gif, filename, step = 1/15):
        ap = StraightAndLevelAutopilot(alt)
        extended_states = True
        res = run_f16_sim(init, tmax, ap, u_seq, step=step, extended_states=extended_states, integrator_str='rk45')

        if gif:
            anim3d.make_anim(res, filename, elev=15, azim=-150, skip_frames=15)    
        return res["times"], res["states"], np.array(res['u_list']), np.array(res['Nz_list'])
    """
    filename = method_name != "none" ? "straight_and_level_$method_name.gif" : "straight_and_level.gif"
    times, states, controls, Nz = py"simulate"(init, alt, u_seq, round(Int64, problem.t_horizon * problem.Δt), gif, filename, step=problem.Δt)

    # only keep the first 13 columns of states (the others are integration variables)
    states = states[:, 1:13]
    for i in 2:9
        states[:, i] = rad2deg(states[:, i])
    end
    # include Nz in states
    # states = hcat(states, Nz)

    # only keep the first 4 columns of controls
    controls = controls[:, 1:4]
    for i in 2:4
        controls[:, i] = rad2deg(controls[:, i])
    end

    Z_actual = Matrix(hcat(states, controls)')
    Z_actual = StatsBase.transform(problem.scaler, Z_actual)

    return times, Z_actual
end

function deg2rad(x)
    return π / 180 * x
end

function rad2deg(x)
    return 180 / π * x
end