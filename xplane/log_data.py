import sys
sys.path.append('..')

from xpc3 import *
import time
import numpy as np

def get_state(client):
        """     
        sim/flightmodel/position/phi (roll)
        sim/flightmodel/position/theta (pitch)
        sim/flightmodel/position/psi (heading)

        Positive P rolls the aircraft to the right; Q pitches the aircraft up; positive R yaws the aircraft to the right.
        sim/flightmodel/position/P (roll rate)
        sim/flightmodel/position/Q (pitch rate)
        sim/flightmodel/position/R (yaw rate)

        sim/flightmodel/forces/vx_acf_axis (velocity in x direction in body axis)
        sim/flightmodel/forces/vy_acf_axis (velocity in y direction in body axis)
        sim/flightmodel/forces/vz_acf_axis (velocity in z direction in body axis)

        sim/flightmodel2/position/beta (sideslip angle)
        sim/flightmodel2/position/alpha (angle of attack)

        Returns [roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot, vx_body, vy_body, vz_body]
        """

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

def get_forces(client):
        """
        """
        # Fx = client.getDREF("sim/flightmodel/forces/fside_aero")[0]
        # Fy = client.getDREF("sim/flightmodel/forces/fnrml_aero")[0]
        # Fz = client.getDREF("sim/flightmodel/forces/faxil_aero")[0]
        # L = client.getDREF("sim/flightmodel/forces/L_aero")[0]
        # M = client.getDREF("sim/flightmodel/forces/M_aero")[0]
        # N = client.getDREF("sim/flightmodel/forces/N_aero")[0]
        
        
        # sim/flightmodel/forces/faxil_total
        # ("sim/flightmodel/forces/fnrml_total")
        # sim/flightmodel/forces/fside_total
        # sim/flightmodel/forces/L_total
        # sim/flightmodel/forces/M_total
        # sim/flightmodel/forces/N_total
        Fx = client.getDREF("sim/flightmodel/forces/fside_total")[0]
        Fy = client.getDREF("sim/flightmodel/forces/fnrml_total")[0]
        Fz = client.getDREF("sim/flightmodel/forces/faxil_total")[0]
        L = client.getDREF("sim/flightmodel/forces/L_total")[0]
        M = client.getDREF("sim/flightmodel/forces/M_total")[0]
        N = client.getDREF("sim/flightmodel/forces/N_total")[0]

        
        # compute dynamic pressure in units of N/(m*m). (dynamic pressure at current flight condition)
        rho = client.getDREF("sim/weather/rho")[0] # density of the air in kg/cubic meters.
        vx = client.getDREF("sim/flightmodel/forces/vx_acf_axis")[0]
        vy = client.getDREF("sim/flightmodel/forces/vy_acf_axis")[0]
        vz = client.getDREF("sim/flightmodel/forces/vz_acf_axis")[0]
        Q = 0.5 * rho * (vx**2 + vy**2 + vz**2)

        # the following two values are for Cessna 172 and are output from X-plane by going to Developer --> Aircraft Performance --> Find Pitch Stability Derivative
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
    """
    Gets the current control inputs. 
    """
    ctrl = client.getCTRL()
    current_elev = ctrl[0]
    current_ail = ctrl[1]
    current_rud = ctrl[2]
    current_throttle = ctrl[3]

    return [current_elev, current_ail, current_rud, current_throttle]


def initialize_aircraft(client):
    heading = 180.0
    initial_posi = [37.46358871459961, -122.11750030517578, 3048.0, 0, 0, heading, 1]  # Lat, Lon, Alt, Pitch, Roll, Yaw, Gear
    client.sendPOSI(initial_posi)        
    speed_mps = 46.3  
    deg2rad = np.pi / 180.0
    Vx = speed_mps * (np.sin(heading*deg2rad))
    Vz = speed_mps * (-np.cos(heading*deg2rad))
    client.sendDREF("sim/flightmodel/position/local_vx", Vx)
    client.sendDREF("sim/flightmodel/position/local_vz", Vz)

def main():
    with XPlaneConnect() as client:
        states = []
        forces = []
        controls = []
        times = []

        start_time = time.time()
        current_time = time.time() - start_time
        duration = 300.0#300.0
        interval = 0.01

        client.pauseSim(True)
        # initialize_aircraft(client)
        client.pauseSim(False)


        # wait for the aircraft to stabilize
        time.sleep(5)

        while current_time < duration:
            if len(states) % 50 == 0:
                print("Time: ", current_time)
                print("Average time per iteration: ", (time.time() - start_time) / (len(states)+1))
            try: 
                state = get_state(client)
                force = get_forces(client)
                control = get_control(client)
            except:
                try: 
                    state = get_state(client)
                    force = get_forces(client)
                    control = get_control(client)
                except:
                    print("Socket timeout")
                    continue
                     
            current_time = time.time() - start_time
            states.append(state)
            forces.append(force)
            controls.append(control)
            times.append(current_time)

            time.sleep(interval)

        # combine all the data into a single numpy array where each row is a time step and each column is a state, force, or control
        states = np.array(states)
        forces = np.array(forces)
        controls = np.array(controls)
        times = np.array(times)

        # save the data to a file
        # np.savez("../data/state_force_control_data_v2.npz", states=states, forces=forces, controls=controls, times=times)
        # np.savez("../data/nonlinear_data.npz", states=states, forces=forces, controls=controls, times=times)
        # np.savez("../data/T38_linear_data.npz", states=states, forces=forces, controls=controls, times=times)
        # np.savez("../data/T38_agressive1_data.npz", states=states, forces=forces, controls=controls, times=times)
        # np.savez("../data/T38_agressive2_data.npz", states=states, forces=forces, controls=controls, times=times)
        # np.savez("../data/T38_mild_aggressive2_data.npz", states=states, forces=forces, controls=controls, times=times)
        # np.savez("../data/short_cessna_data.npz", states=states, forces=forces, controls=controls, times=times)
        np.savez("../data/T38_total_state_force_control_data.npz", states=states, forces=forces, controls=controls, times=times)




if __name__ == "__main__":
    main()
