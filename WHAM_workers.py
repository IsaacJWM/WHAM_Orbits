import particle_sieve as ps
import taylor_field_tools as tft
import WHAMField
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import maxwell
from scipy.stats import uniform_direction
import os
import h5py
import matplotlib.pyplot as plt
import time

sys.path.insert(0, "./classes")

import mem_eff_particle as pt
    

def new_run_particle(initial_pos, initial_vel, bFunc, vertices, norbits=10, dt=0.1, 
                     dump_size=10000, write_data=True):

    p1 = pt.mem_eff_particle(initial_pos, initial_vel, dt, int(norbits * 2 * np.pi / dt), dump_size=dump_size,
                     write_data=write_data, silent=True)
                    # creating the particle with the given conditions
    p1.set_boundaries(vertices=vertices)
    p1.step(bFunc)  #subtract 1 to get nice # of iterations instead of nice # of orbits

    return p1


def new_run_position(position,bFunc,vertices,norbits=100,dt=0.01,no_chunks=True,filename=None):
    """
    Takes a single input position (x,y,z) in grid units and runs particles
    at NVEL velocities sampled from a normal distribution
    """

    xloc,yloc,zloc = position
    
    vx = ps.select_velocities(1)
    vy = ps.select_velocities(1)
    vz = ps.select_velocities(1)
    
    if no_chunks:
        dump_size = norbits/dt
    else:
        dump_size = None
    
    p1 = new_run_particle([xloc,yloc,zloc], [vx[0],vy[0],vz[0]], bFunc, vertices, 
                          norbits=norbits, dt=dt, dump_size=dump_size, write_data=False)

    #print("Approximate run time is {:d} velocities x {:e} iterations x {:1.4f} seconds per iteration = {:4.2f}".format(nvel, norbits/dt, p1.iter_time, nvel*norbits*p1.iter_time/dt))
    
    if p1.outOfBounds:
        filename = filename + "_escaped.h5"
    else:
        filename = filename + "_confined.h5"
        
    return p1, filename, vx[0]


def RunGrid(norbits, nvel, vertices, dt=0.1, m=1, q=1, T=1, B0=1, scale=1, 
                    shaper=(0,0.4), shapez=(-1,1), filename='data//WHAMTest//Memory_test//'):
    
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    shaper *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    shapez *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    bufferr = shaper[1] / 10
    bufferz = shapez[1] / 10
    rr = np.repeat(np.linspace(shaper[0]+bufferr, shaper[1]-bufferr, 4, endpoint=True), nvel)
    zz = np.repeat(np.linspace(shapez[0]+bufferz, shapez[1]-bufferz, 8, endpoint=True), nvel)
    
    vertices *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    
    #for r in rr:
    #    for z in zz:
    #        new_run_position([0,r,z], field_data.field, vertices, norbits, dt, 
    #                         True, filename+'_r'+str(round(r))+'_z'+str(round(z)))
    
    all_args = [
        ([0, rloc, zloc], field_data.field, vertices, norbits, dt, 
         True,filename+'_r'+str(round(rloc))+'_z'+str(round(zloc)))
        for zloc in zz
        for rloc in rr
        ]
    
    for nw in [4, 8, 16]:
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=nw) as executor:
            futures = {executor.submit(new_run_position, *args): args for args in all_args}
            for future in as_completed(futures):
                try:
                    particle, filename, v = future.result()  # this re-raises the actual exception from the worker
                except Exception as e:
                    print(f"Worker failed with: {type(e).__name__}: {e}")
                    continue
                #ps.write_single_position_data(particle,filename,f"v{v:03.3f}",write_mode='a')
                del particle
        elapsed = time.perf_counter() - start_time
        print(f"{nw} workers: {elapsed:.1f}s")
        
def plot_z_vs_t(file_path):
    file = file_path.split("/")[-1]
    with h5py.File(file_path, mode='r') as ff:
        for i, v in enumerate(ff.keys()):
            plt.plot(list(range(len(ff[v]["r"]))), ff[v]['r'][:,2])
            plt.xlabel("Iteration")
            plt.ylabel("Z-position")
            plt.title(f"File {file}, Particle number {i}")
            plt.show()
            
def plot_trajectory(file_path):
    file = file_path.split("/")[-1]
    with h5py.File(file_path, mode='r') as ff:
        for i, v in enumerate(ff.keys()):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*ff[v]['r'][:].T, c=np.arange(len(ff[v]['r'])), cmap='viridis', s=2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.title(f"File {file}, Particle number {i}")
            plt.show()

def plot_zs_vs_t(directory, confined=True):
    files = os.listdir(directory)
    for file in files:
        if (file.endswith("_confined.h5") and confined) or (file.endswith("_escaped.h5") and not confined):
            with h5py.File(os.path.join(directory,file), mode='r') as ff:
                for i, v in enumerate(ff.keys()):
                    plt.plot(list(range(len(ff[v]["r"]))), ff[v]['r'][:,2])
    plt.xlabel("Iteration")
    plt.ylabel("Z-position")
    plt.title("Particle positions vs time")
    plt.show()

def plot_trajectories(directory, confined=True):
    files = os.listdir(directory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for file in files:
        if (file.endswith("_confined.h5") and confined) or (file.endswith("_escaped.h5") and not confined):
            with h5py.File(os.path.join(directory,file), mode='r') as ff:
                for i, v in enumerate(ff.keys()):
                    ax.plot(*ff[v]['r'][:].T)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Particle trajectories")
    plt.show()

def get_fraction_lost(directory):
    files = os.listdir(directory)
    confined = 0
    escaped = 0
    for file in files:
        if file.endswith('_confined.h5'):
            confined += 1
        elif file.endswith('_escaped.h5'):
            escaped += 1
    frac_confined = confined / (confined + escaped)
    print(f"Total confined: {confined}")
    print(f"Total escaped: {escaped}")
    print(f"Fraction confined: {frac_confined}")


