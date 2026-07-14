import WHAM_workersv02 as workers
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
    
    workers.RunGrid(norbits=10000, nvel=3, vertices=V, dt=0.1, m=m, q=q, T=T, B0=B0, scale=scale, 
                    shaper=np.array([1e-10,0.15]), shapez=np.array([-0.7,0.7]), filepath=directory)
    
    conf, esc = workers.read_data(os.path.join(directory, "output.pkl"))
    
    workers.get_fraction_lost(conf, esc)
    workers.confined_in_vperp_vpar_space(conf, esc, savedir="./output")
    workers.confinement_over_time(conf, esc, savedir="./output")
    
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    workers.plot_confinement_with_fieldlines(conf, esc, field_data.field, scale=(scale/0.000102)*np.sqrt(m*T) / (q*B0), savedir="./output")
    
    workers.plot_confined_by_pitch_angle(conf, esc)
    
    #workers.plot_3d_fieldlines(field_data.field, scale=(scale/0.000102)*np.sqrt(m*T) / (q*B0))


























