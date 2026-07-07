import WHAM_workers as workers
import orbit_statistics
import numpy as np
import os
import h5py


if __name__ == "__main__":
    
    V = np.array([[0,-1], [0.0557, -1], [0.0557, -0.776], [0.2, -0.776], 
        [0.2, 0.776], [0.0557, 0.776], [0.0557, 1], [0, 1]])
    workers.RunGrid(norbits=10000, nvel=1, vertices=V, dt=0.1, m=1.0, q=1.0, T=1.0, B0=1.0, scale=1.0, 
                    shaper=np.array([0.05,0.15]), shapez=np.array([-0.75,0.75]))


    
    #confined = orbit_statistics.read_single_position_files("data\\WHAMTest", save_output=False)

    #escaped = orbit_statistics.read_single_position_files("data\\WHAMTest", read_escaped=True, save_output=False)

    #path = r"C:\Users\Student\Desktop\Particle_Orbits\Orbits\data\WHAMTest"
    #file_list = os.listdir(path)

    #confined_list = [f for f in file_list if f.endswith("_confined.h5")]
    #escaped_list = [f for f in file_list if f.endswith("_escaped.h5")]
    
    #with h5py.File("data/WHAMTest/Test_1_r637_z-2206_confined.h5") as ff:
    #    vels = ff.keys()
    #    print(vels)
    #    for v in vels:
    #        print(ff[v]['r'][:,2])
    #        print(range(ff[v]['iter'][0]))
    
    
    #workers.plot_z_vs_t("data/WHAMTest/Test_1_r637_z-2206_confined.h5")
    #workers.plot_trajectory("data/WHAMTest/Test_1_r637_z-2206_confined.h5")
    #workers.plot_zs_vs_t("data\\WHAMTest", confined=False)
    #workers.plot_trajectories("data\\WHAMTest", confined=False)

    



























