import WHAM_workers as workers
import orbit_statistics
import numpy as np
import os
import h5py
import tracemalloc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    V = np.array([[0,-1], [0.0557, -1], [0.0557, -0.776], [0.2, -0.776], 
        [0.2, 0.776], [0.0557, 0.776], [0.0557, 1], [0, 1]])
    workers.RunGrid(norbits=10000, nvel=5, vertices=V, dt=0.1, m=1.0, q=1.0, T=1.0, B0=1.0, scale=1.0, 
                    shaper=np.array([0.05,0.15]), shapez=np.array([-0.75,0.75]))
    


























