import WHAM_workers as workers
import numpy as np
import os
import h5py
import orbit_statistics
import WHAMField


if __name__ == "__main__":
    
    m = 2
    q = 1
    B0 = 1
    T = 50
    scale = 1
    
    
    V = np.array([[0,-1], [0.0557, -1], [0.0557, -0.776], [0.2, -0.776], 
        [0.2, 0.776], [0.0557, 0.776], [0.0557, 1], [0, 1]])
    workers.RunGrid(norbits=100, nvel=5, vertices=V, dt=0.1, m=m, q=q, T=T, B0=B0, scale=scale, 
                    shaper=np.array([1e-10,0.15]), shapez=np.array([-0.7,0.7]))
    
    #directory = "C:/Users/Student/Desktop/Particle_Orbits/Orbits/data/WHAMTest/Memory_test"
    
    #workers.get_fraction_lost(directory)
    #workers.confined_in_vperp_vpar_space(directory)
    #workers.confinement_over_time(directory)
    
    #field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    #workers.plot_confinement_with_fieldlines(directory, field_data.field)


























