import WHAM_workersv02 as workers
#import WHAM_workers as old_workers
#import particle_sieve as ps
import numpy as np
import WHAMField
import os


if __name__ == "__main__":
    #Testing git push from new computer again
    
    
    
    directory = "./data/Firebird_runs/"
    
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    m = 2
    q = 1
    B0 = 1
    T = 50
    scale = 1
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    V = np.array([[0,-1], [0.0557, -1], [0.0557, -0.776], [0.2, -0.776], 
        [0.2, 0.776], [0.0557, 0.776], [0.0557, 1], [0, 1]])
    
    #particle, fname, v = old_workers.run_particle_in_grid([100, 100, 0], field_data.field, V, 10000, 1, filename="./data/WHAMTest/Troubleshooting")
    #ps.write_single_position_data(particle,fname,f"v{v:03.3f}",write_mode='a')
    
    #old_workers.plot_z_vs_t(fname, savedir="./output/")
    #old_workers.plot_trajectory(fname, savedir="./output/")
    
    print("Function starting")
    workers.RunGrid(norbits=100000, nvel=10000, vertices=V, dt=1, m=m, q=q, T=T, B0=B0, scale=scale, 
                    shaper=np.array([1e-10,0.15]), shapez=np.array([-0.25,0.25]), filepath=directory)
    print("Function closed")
    
    V = V * (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    """
    particle, fname, v = workers.run_particle_in_grid([1, 0, 0], field_data.field, V*(scale/0.000102) *np.sqrt(m*T) / (q*B0), 10, 1)
    r = particle.get_r()
    print(particle.get_B())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Test")
    plt.show()
    """
    

    conf, esc = workers.read_data(os.path.join(directory, "output.pkl"))
    
    workers.plot_escaped_positions_2d(esc, V, field_data.field, scale=(scale/0.000102)*np.sqrt(m*T), savedir="./output/")
    
    workers.get_fraction_lost(conf, esc)
    workers.confined_in_vperp_vpar_space(conf, esc, savedir="./output/")
    workers.confinement_over_time(conf, esc, savedir="./output/")
    workers.plot_confinement_with_fieldlines(conf, esc, field_data.field, scale=(scale/0.000102)*np.sqrt(m*T) / (q*B0), savedir="./output/")
    workers.plot_confined_by_pitch_angle(conf, esc, savedir="./output/")
    
    #workers.plot_3d_fieldlines(field_data.field, scale=(scale/0.000102)*np.sqrt(m*T) / (q*B0))























