import WHAM_workers as workers
import numpy as np
import WHAMField
import os


if __name__ == "__main__":
    
    directory = "./data/Firebird_runs/"
    
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    m = 2
    q = 1
    B0 = 1
    T = 50
    scale = 1
    
    V = np.array([[0,-1], [0.0557, -1], [0.0557, -0.776], [0.2, -0.776], 
        [0.2, 0.776], [0.0557, 0.776], [0.0557, 1], [0, 1]])
    
    workers.RunGrid(norbits=100, nvel=10, vertices=V, dt=0.1, m=m, q=q, T=T, B0=B0, scale=scale, 
                    shaper=np.array([1e-10,0.15]), shapez=np.array([-0.7,0.7]), filename=directory)
    
    files = os.listdir(directory)
    for file in files[:10]:
        workers.plot_z_vs_t(os.path.join(directory, file), savedir="./output")
        workers.plot_trajectory(os.path.join(directory, file), savedir="./output")
    
    workers.get_fraction_lost(directory)
    workers.confined_in_vperp_vpar_space(directory, savedir="./output")
    workers.confinement_over_time(directory, savedir="./output")
    
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    workers.plot_confinement_with_fieldlines(directory, field_data.field, savedir="./output")
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


























