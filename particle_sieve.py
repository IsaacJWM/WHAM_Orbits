from taylor_field_tools import run_particle
from taylor_field_tools import load_taylor_field
import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

## Script for running particle confinement sieve

def set_up_grid(nx=8, ny=8, nz=8, xmin=-95, xmax=95, zmin=10, zmax=500, show_plot=False):
    ## Reminder: PSI-Tet mesh is cylinder with radius 1 and length 10, one end at origin

    ## Continue using particle grid units of 100x100x1000 (a la Hari),
    ## corresponding to a 100:1 discretization of the volume.

    ## create 1D arrays spanning the bounding rectangular prism
    x = np.linspace(xmin,xmax,num=nx,endpoint=True)
    y = np.linspace(xmin,xmax,num=ny,endpoint=True)
    z = np.linspace(zmin,zmax,num=nz,endpoint=True)

    ## create 3D mesh from 1D spans
    xx,yy,zz = np.meshgrid(x,y,z)

    ## Remove points outside radius of cylinder
    rr = np.sqrt(xx**2 + yy**2)

    keep_indices = np.where(rr <= 100.0)

    if show_plot:
        ax = plt.gcf().add_subplot(111,projection='3d')
        ax.plot(xx[keep_indices],yy[keep_indices],zz[keep_indices],'o')
        ax.set_xlim(-z.max()/2.,z.max()/2.)
        ax.set_ylim(-z.max()/2.,z.max()/2.)
        ax.set_zlim(z.min(),z.max())
        plt.show()

    return xx[keep_indices].astype(int),yy[keep_indices].astype(int),zz[keep_indices].astype(int)

def select_velocities(nvel=100, seed=None):
    ##Choose a sample of velocities from a normal distribution
    ## Optionally, specify the seed to get repeatable test cases

    if seed is not None:
        np.random.seed(seed)

    return np.random.randn(nvel)


def run_position(position,norbits=100,nvel=100,dt=0.01,test=False,
                no_chunks=True,filename=None):
    """
    Takes a single input position (x,y,z) in grid units and runs particles
    at NVEL velocities sampled from a normal distribution
    """

    if filename is None:
        filename = "trajectories_x{:02d}y{:02d}z{:03d}_unlabeled".format(position[0],
                                                           position[1],
                                                           position[2])

    field_data = load_taylor_field()

    #Run particles at each of NVEL velocities

    print("\nRunning particles starting from position: "+str(position))

    xloc,yloc,zloc = position

    xvel = select_velocities(nvel=nvel)
    yvel = select_velocities(nvel=nvel)
    zvel = select_velocities(nvel=nvel)

    velocities = zip(xvel,yvel,zvel)

    #Run particle starting at POSITION at each of the initial velocities
    for ii,v0 in enumerate(velocities):
        if no_chunks:
            dump_size = norbits/dt
        else:
            dump_size = None

        p1 = run_particle([xloc,yloc,zloc],[v0[0],v0[1],v0[2]],
             field_data=field_data,norbits=norbits,dt=dt,
             dump_size=dump_size,write_data=False)

        print("Approximate run time is {:d} velocities x {:e} iterations x {:1.4f} seconds per iteration = {:4.2f}".format(nvel, norbits/dt, p1.iter_time, nvel*norbits*p1.iter_time/dt))

        if p1.outOfBounds:
            write_single_position_data(p1,filename+"_escaped.h5",
                                       "v{:002d}".format(ii),write_mode='a')
        else:
            write_single_position_data(p1,filename+"_confined.h5",
                                       "v{:002d}".format(ii),write_mode='a')

    return list(velocities)

def write_single_position_data(p1,filename,groupname,write_mode='w-'):

    try:
        with h5py.File(filename,write_mode) as hf:
            #create a new group if it doesn't already exist
            while groupname in hf.keys():
                print("Duplicate group name: "+groupname)
                groupname = 'v' + str(float(groupname[1:]) + 0.001)
            gp = hf.create_group(groupname)

            gp.create_dataset('r',data=p1.r[:-1])
            gp.create_dataset('v',data=p1.v[:-1])
            gp.create_dataset('B',data=p1.Bfield[:-1])
            # subtract 1 from the shape for initial condition

            gp.create_dataset('iter', data=[p1.iter])
            gp.create_dataset('outOfBounds', data=[p1.outOfBounds])
            gp.create_dataset('dt', data=[p1.dt])

    except IOError as fileerr:
        # for debugging
        print(fileerr)

        #if the file already exists, ask if you want to overwrite
        print("\nThis data file already exists.")
        user_input = input("Do you wish to overwrite it (o), append to it (a), or cancel data dump (c)? (o/a/c)")

        if user_input == 'o':
            write_single_position_data(p1,filename,groupname,write_mode='w')
            print("Data file will be overwritten.")
        elif user_input == 'a':
            write_single_position_data(p1,filename,groupname,write_mode='a')
            print("Data will be appended to file.")
        else:
            sys.exit("Canceling data dump.\n")
            pass




def test_single_position_run(filename,nvel):
    """
    Plots trajectories for all velocities given a single-position save file.
    """

    ax = plt.gcf().add_subplot(111,projection='3d')

    with h5py.File(filename,'r') as ff:

        for v_number in range(nvel):
            gp = ff['v{:02d}'.format(v_number)]
            ax.plot(gp['r'][:,0],gp['r'][:,1],gp['r'][:,2])

    return ax

def run_sieve(norbits=100,dt=0.01,test=False,no_chunks=True,
              filename='trajectory_sieve_unlabeled.h5'):

    field_data = load_taylor_field()

    xx,yy,zz = set_up_grid()

    #Run particles in each plane
    for zindex,zloc in enumerate(zz):

        if test:
            #only run the first plane
            if zindex>1:
                break

        print("Running particles in plane: z=%2.1f"%zloc)

        for yindex,yloc in enumerate(yy):
            for xindex,xloc in enumerate(xx):

                xvel = select_velocities()
                yvel = select_velocities()
                zvel = select_velocities()

                velocities = zip(xvel,yvel,zvel)

                groupname = 'x'+xloc+'_y'+yloc+'_z'+zloc
                nvelocities = xvel.shape[0]

                #Run each position at each of the initial velocities
                for v0 in velocities:
                    if no_chunks:
                        dump_size = norbits/dt
                    else:
                        dump_size = None

                    p1 = run_particle([xloc,yloc,zloc],[v0[0],v0[1],v0[2]],
                         field_data=field_data,norbits=norbits,dt=dt,
                         dump_size=dump_size,write_data=False)

                    if p1.outOfBounds:
                        continue
                    else:
                        write_data(p1,filename,groupname,nvelocities)


def write_all_data(p1,filename,groupname):

    with h5py.File(filename,'w-') as hf:
        #create a new group if it doesn't already exist
        try:
            hf.create_group(groupname)
            group_exists = False
        except ValueError:
            group_exists = True

        #labe group so we can create trajectories inside of it
        gp = hf[groupname]

        if group_exists:
            pass
            #DO WE NEED TO DO SOMETHING ACTIVE FOR APPENDING?




        # subtract 1 from the shape for initial condition
        # each vector has dimensions (nvelocities,nsteps,3)
        write_length = np.asarray(self.r).shape[0]
        rdata = gp.create_dataset("r", (write_length-1,3), maxshape=(None, 3))
        rdata[:,:] = self.r[:-1]
        vdata = gp.create_dataset("v", (write_length-1,3), maxshape=(None, 3))
        vdata[:,:] = self.v[:-1]

        Bdata = gp.create_dataset("B", (write_length-1,3), maxshape=(None, 3))
        Bdata[:,:] = self.Bfield[:-1]

        hf.create_dataset('iter', data=[self.iter])
        hf.create_dataset('outOfBounds', data=[self.outOfBounds])
        hf.create_dataset('dt', data=[self.dt])


# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
if __name__ == '__main__':
    run_sieve()
