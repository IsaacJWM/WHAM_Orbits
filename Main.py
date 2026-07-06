import WHAM_workers as workers
import orbit_statistics
import numpy as np



if __name__ == "__main__":
    
    #V = np.array([[0,-1], [0.0557, -1], [0.0557, -0.776], [0.2, -0.776], 
    #    [0.2, 0.776], [0.0557, 0.776], [0.0557, 1], [0, 1]])
    #workers.RunGrid(norbits=1000, nvel=4, vertices=V, dt=0.1, m=1.0, q=1.0, T=1.0, B0=1.0, scale=1.0, 
    #                shaper=np.array([0.05,0.15]), shapez=np.array([-0.75,0.75]))

    ipos, fpos, ivel, iB, maxit, conf = orbit_statistics.read_single_position_files("data\\WHAMTest", save_output=False)


    print("Confined:")
    print(ipos)
    print(fpos)
    print(ivel)
    print(maxit)
    print(conf)

    print("Escaped:")

    ipos, fpos, ivel, iB, maxit, conf = orbit_statistics.read_single_position_files("data\\WHAMTest", read_escaped=True, save_output=False)
    print(ipos)
    print(fpos)
    print(ivel)
    print(maxit)
    print(conf)


    


































