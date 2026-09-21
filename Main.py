import Workers as workers
import Trajectories as trajectories
#import WHAM_workers as old_workers
#import particle_sieve as ps
import numpy as np
import WHAMField
import os
import classes.Fields as Fields
import particle_sieve as ps


if __name__ == "__main__":
    
    
    # Normalization parameters. Modify these to adjust particle velocities (T), mass (m), or charge (q)
    m = 2
    q = 1
    B0 = 1
    T = 100
    scale = 1
    
    # Normalization function for distances. x is an integer or a list representing distance in meters.
    def normalize(x):
        x *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
        return x
    
    # Defining the directory where data will be stored.
    directory = "./data/Trajectories/"
    #directory = "./data/Runs/"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    # Loading the magnetic field
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    # Defining the boundary of WHAM in r-z space
    V = np.array([[0,-1], [0.0557, -1], [0.0557, -0.776], [0.2, -0.776], 
        [0.2, 0.776], [0.0557, 0.776], [0.0557, 1], [0, 1]])
    V = normalize(V)
    
    
    
    # Getting plots for poster
    """
    ss = np.random.SeedSequence()
    #p = trajectories.run_particle([10**(-15), 0, 1], Fields.getWireField, [[0, -1000], [1000, -1000], [1000, 1000], [0, 1000]], norbits=5, seed=ss.spawn(1)[0])
    #ps.write_single_position_data(p,os.path.join(directory, "Wire_field.h5"),'1',write_mode='a')
    trajectories.plot_z_vs_x(os.path.join(directory, "Wire_field.h5"), directory)
    trajectories.plot_trajectory(os.path.join(directory, "Wire_field.h5"), directory)
    """
    
    #=================== Thermal run ===================#
    """
    # Defining the boundaries of the grid of starting positions.
    shaper = np.array([1e-10,0.15])
    shaper = normalize(shaper)
    shapez = np.array([-0.25,0.25])
    shapez = normalize(shapez)
    
    # Defining the number of grid points in the r and z directions
    nr = 10
    nz = 10
    
    # Defining the length of time each particle will run for
    norbits = 100000
    
    # Defining the number of particles to run at each grid point
    nvel = 10000
    
    # Defining the time step between calculations
    dt = 0.1
    
    # Function call for large runs, does not save trajectories
    #workers.RunGrid(nr=nr, nz=nz, nvel=nvel, norbits=norbits, dt=dt, field=field_data.field,
    #                vertices=V, shaper=np.array([1e-10,0.15]), shapez=np.array([-0.25,0.25]), 
    #                filepath=directory)
    
    # Function call to save trajectories. ONLY USE FOR SMALL RUNS
    trajectories.RunGrid(nr=nr, nz=nz, nvel=nvel, norbits=norbits, dt=dt, field=field_data.field,
                    vertices=V, shaper=np.array([1e-10,0.15]), shapez=np.array([-0.25,0.25]), 
                    filepath=directory)
    """
    
    
    #=================== NBI run ===================#
    """
    # Number of particles to calculate
    nparticles = 100000
    
    # Number of orbits to run for each particle
    norbits = 100000
    
    # Timestep between calculations
    dt = 1
    
    # Beam velocity (as a multiple of thermal velocity T) and beam direction
    beam_mag = 10
    beam_dir = np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)])
    
    # Beam characteristics
    # Mean free path of neutral beam particles in the plasma
    mfp = 0.1
    mfp = normalize(mfp)
    # Position at which beam enters the plasma
    ipos = np.array([-0.2, 0, -0.2])
    ipos = normalize(ipos)
    # radius of the beam
    rmax = 0.05
    rmax = normalize(rmax)
    
    
    # Function call for large runs, does not save trajectories
    workers.RunNBI(nparticles=nparticles, norbits=norbits, dt=dt, field=field_data.field, vertices=V,
        beam_v=beam_mag, vdir=beam_dir, mfp=mfp, ipos=ipos, rmax=rmax, filepath=directory)
    
    # Function call to save trajectories. ONLY USE FOR SMALL RUNS
    trajectories.RunNBI(100000, 10, V, dt=0.1, m=m, q=q, T=T, B0=B0, scale=scale, v=10, vdir=np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)]),
        mfp=0.1, ipos=np.array([-0.2, 0, -0.2]), rmax=0.05, filepath=directory)
    """
    
    
    #for file in os.listdir(directory):
    #    trajectories.plot_trajectory(os.path.join(directory,file))
    #trajectories.plot_trajectories(directory, confined=True)
    
    
    #conf, esc = workers.read_data(os.path.join(directory, "NBI_output.pkl"))
    #workers.plot_initial_positions(conf, esc, V)
    #workers.plot_escaped_positions_2d(esc, V, field_data.field, scale=(scale/0.000102)*np.sqrt(m*T), savedir="./output/")
    
    #workers.get_fraction_lost(conf, esc)
    #workers.confined_in_vperp_vpar_space(conf, esc, savedir="./output/")
    #workers.confinement_over_time(conf, esc, savedir="./output/")
    #workers.plot_confinement_with_fieldlines(conf, esc, field_data.field, scale=(scale/0.000102)*np.sqrt(m*T) / (q*B0), savedir="./output/")
    #workers.plot_confined_by_pitch_angle(conf, esc, savedir="./output/")
    
    #workers.plot_3d_fieldlines(field_data.field, scale=(scale/0.000102)*np.sqrt(m*T) / (q*B0))

    

















