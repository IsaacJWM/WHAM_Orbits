import particle_sieve as ps
import taylor_field_tools as tft
import WHAMField
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import maxwell
from scipy.stats import uniform_direction
import os
import h5py
import matplotlib.pyplot as plt

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
    
    vmag = maxwell.rvs(scale=np.sqrt(np.pi/8),size=nvel)
    vhat = uniform_direction.rvs(3,nvel)
    velocities = vhat * vmag[:, np.newaxis]

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


def RunGrid(norbits, nvel, vertices, dt=0.1, m=1, q=1, T=1, B0=1, scale=1, 
                    shaper=(0,0.4), shapez=(-1,1), filename='data//WHAMTest//Test_1'):
    
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    shaper *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    shapez *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    bufferr = shaper[1] / 10
    bufferz = shapez[1] / 10
    rr = np.linspace(shaper[0]+bufferr, shaper[1]-bufferr, 4, endpoint=True)
    zz = np.linspace(shapez[0]+bufferz, shapez[1]-bufferz, 8, endpoint=True)
    
    vertices *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)

    #for r in rr:
    #    for z in zz:
    #        new_run_position([0,r,z], field_data.field, vertices, norbits, nvel, dt, 
    #                         True, filename+'_r'+str(round(r))+'_z'+str(round(z)))
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


def plot_z_vs_t(file_path):
    file = file_path.split("/")[-1]
    with h5py.File(file_path, mode='r') as ff:
        for i, v in enumerate(ff.keys()):
            plt.plot(list(range(ff[v]["iter"][0])), ff[v]['r'][:,2])
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
            ax.scatter(*ff[v]['r'][:].T, c=np.arange(ff[v]['iter'][0]), cmap='viridis', s=2)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.title(f"File {file}, Particle number {i}")
            plt.show()




def plot_trajectories(file_list, velocities_by_file = None,
    data_directory = '/Volumes/alight/trajectory_merge_6524539_7294312_10186475/', image_directory='/Users/alight/Dropbox/python/particle_orbits/data/trajectory_images/',
    image_folder_prefix=None, image_subdir='default_subdir',
    v_min=0.0, save_plots=True):
    """
    Takes list of filenames and initial velocity values for a set of
    particles and plots the trajectory corresponding to each.

    By default all velocity indices in each file are plotted.

    To choose specific velocity indices in each file, VELOCITIES_BY_FILE
    is provided as a list of velocities for each vile in FILE_LIST.

    """


    for ii, datafile in enumerate(file_list):

        fname = datafile.split('/')[-1]
        if image_folder_prefix is not None:
            image_subdir = os.path.join(image_directory,image_folder_prefix+fname[13:25])
            try:
                os.mkdir(image_subdir)
            except FileExistsError:
                pass
        else:
            pass


        with h5py.File(os.path.join(data_directory,datafile), mode='r') as ff:
            if velocities_by_file is None:
                vi_list = ff.keys()
            else:
                vi_list = velocities_by_file[ii]

            for vlabel in vi_list:
                v_mag = np.sqrt((ff[vlabel]['v'][0]**2).sum())
                if v_mag < v_min:
                    continue

                #set up axis
                fig = plt.gcf()
                fig.clf()
                ax = fig.add_subplot(111, projection='3d')

                zmin = ff[vlabel]['r'][()][:,2].min()
                zmax = ff[vlabel]['r'][()][:,2].max()
                ax.set_zlim(0.9*zmin,1.1*zmax)
                tft.generate_cylinder((0, 0, zmin), (0, 0, zmax), 100, ax=ax)
                ax.set_title(fname+", "+vlabel+"\nSpeed [v_th] = {:.2f}".format(v_mag), y=1.02)

                #plot trajectory and save image
                tft.time_as_color_3D(ff[vlabel]['r'][()],ax=ax)
                if save_plots:
                    plt.savefig(os.path.join(image_directory,image_subdir,fname.split('.')[0]+'_'+vlabel+'.png'))
    return ax
