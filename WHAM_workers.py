import particle_sieve as ps
#import taylor_field_tools as tft
import Jack_Code.WHAMField as WHAMField
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "./classes")

import Particle as pt

def WHAMGrid(nr=3, nz=10, rmin=0, rmax=1000, zmin=-2000, zmax=2000):
    
    r = np.linspace(rmin, rmax, num=nr, endpoint=True)
    z = np.linspace(zmin,zmax,num=nz,endpoint=True)
    
    rr, zz = np.meshgrid(r, z)
    
    return rr.astype(int), zz.astype(int)
    

def new_run_particle(initial_pos, initial_vel, bFunc, vertices, norbits=10, dt=0.1, 
                     dump_size=10000, data_dump_path='./', write_data=True):

    p1 = pt.particle(initial_pos, initial_vel, dt, dump_size=dump_size,
                    data_dump_path=data_dump_path,write_data=write_data, silent=True)
                    # creating the particle with the given conditions
    p1.set_boundaries(vertices=vertices)
    p1.step(bFunc, int(norbits * 2 * np.pi / p1.dt)-1)  #subtract 1 to get nice # of iterations instead of nice # of orbits

    return p1


def new_run_position(position,bFunc,vertices,norbits=100,nvel=100,dt=0.01,no_chunks=True,filename=None):
    """
    Takes a single input position (x,y,z) in grid units and runs particles
    at NVEL velocities sampled from a normal distribution
    """

    xloc,yloc,zloc = position

    xvel = ps.select_velocities(nvel=nvel)
    yvel = ps.select_velocities(nvel=nvel)
    zvel = ps.select_velocities(nvel=nvel)

    velocities = zip(xvel,yvel,zvel)

    #Run particle starting at POSITION at each of the initial velocities
    for ii,v0 in enumerate(velocities):
        if no_chunks:
            dump_size = norbits/dt
        else:
            dump_size = None
        
        p1 = new_run_particle([xloc,yloc,zloc], [v0[0],v0[1],v0[2]], bFunc, vertices, 
                              norbits=norbits, dt=dt, dump_size=dump_size, write_data=False)

        #print("Approximate run time is {:d} velocities x {:e} iterations x {:1.4f} seconds per iteration = {:4.2f}".format(nvel, norbits/dt, p1.iter_time, nvel*norbits*p1.iter_time/dt))

        if p1.outOfBounds:
            ps.write_single_position_data(p1,filename+"_escaped.h5",
                                       "v{:002d}".format(ii),write_mode='a')
        else:
            ps.write_single_position_data(p1,filename+"_confined.h5",
                                       "v{:002d}".format(ii),write_mode='a')

    return list(velocities)


def RunTrajectories(norbits, nvel, vertices, dt=0.1, m=1, q=1, T=1, B0=1, scale=1, 
                    shaper=(0,0.4), shapez=(-1,1), filename='data//WHAMTest//Test_1'):
    
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    shaper *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    shapez *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    bufferr = shaper[1] / 10
    bufferz = shapez[1] / 10
    rr = np.linspace(shaper[0]+bufferr, shaper[1]-bufferr, 5, endpoint=True)
    zz = np.linspace(shapez[0]+bufferz, shapez[1]-bufferz, 10, endpoint=True)

    # Build a flat list of all (rloc, zloc) combinations
    all_args = [
        ([0, rloc, zloc], field_data.field, vertices, norbits, nvel, dt, 
         True,filename+'_r'+str(round(rloc))+'_z'+str(round(zloc)))
        for zloc in zz
        for rloc in rr
        ]
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(new_run_position, *args) for args in all_args]
        for future in futures:
            try:
                future.result()  # this re-raises the actual exception from the worker
            except Exception as e:
                print(f"Worker failed with: {type(e).__name__}: {e}")
                break