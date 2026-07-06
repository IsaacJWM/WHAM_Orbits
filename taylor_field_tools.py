import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import sys
import time
import os

from datetime import datetime
from pathlib import Path
from scipy.linalg import norm
from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d.art3d as art3d
from scipy.ndimage import median_filter, gaussian_filter1d

sys.path.insert(0, "./classes")

import Particle as pt
import Fields
import particle_sieve as pr

import mytools
#import marklin as mktools

# manually importing data sets
#trajectory_data_path = "/Users/adamdlight/SSX_data/particle_orbits/Taylor10t1/taylor_volume/"
#field_data_path = '/Users/alight/Dropbox/python/particle_orbits/data/cyl10_bfield_xyz.dat'
#field_data_path = 'C:\\Users\\Adam\\Dropbox\\python\\particle_orbits\\data\\cyl10_bfield_xyz.dat'
#data_dump_path = '/Users/alight/Dropbox/python/particle_orbits/data/'
#data_dump_path = 'C:\\Users\\Adam\\Dropbox\\python\\particle_orbits\\data\\'
#field_data_path = os.path.join(Path.home(), 'Dropbox', 'python', 'particle_orbits', 'data', 'cyl10_bfield_xyz.dat')
#data_dump_path = os.path.join(Path.home(), 'Dropbox', 'python', 'particle_orbits', 'data')
field_data_path = os.path.realpath("./data/cyl10_bfield_xyz.dat")
data_dump_path = os.path.realpath("./data/")

def load_taylor_field(filename=field_data_path, normalized=True):
    start = time.time()

    sizes = (72, 72, 200)
    data = np.loadtxt(filename, skiprows=3)

    x = data[:, 0].reshape(sizes)  # putting the data in arrays based on location in file
    y = data[:, 1].reshape(sizes)
    z = data[:, 2].reshape(sizes)

    cached_bx = data[:, 3].reshape(sizes)
    cached_by = data[:, 4].reshape(sizes)
    cached_bz = data[:, 5].reshape(sizes)

    cached_bx[np.where(np.abs(cached_bx) > 1e10)] = 0.0  # a different method of setting to 0 as in particle.py
    cached_by[np.where(np.abs(cached_by) > 1e10)] = 0.0
    cached_bz[np.where(np.abs(cached_bz) > 1e10)] = 0.0

    if normalized:
        """
        Need for consistency in setting B0.

        Right now, B0 is 1./modB.max() for Hari's bridges runs.
        """
        modb = np.sqrt(cached_bx ** 2 + cached_by ** 2 + cached_bz ** 2)
        cached_bx /= modb.max()
        cached_by /= modb.max()
        cached_bz /= modb.max()

    print(("Field data load time: " + str(time.time() - start)))  # simply gives us the time the function took to run

    return x, y, z, cached_bx, cached_by, cached_bz

def get_positions(test=False, write_data=False):
    if test:
        #just two grid points for a short test
        xx,yy,zz = pr.set_up_grid(nx=2,ny=1,nz=1,xmin=50,xmax=60,zmin=300,zmax=500)
    else:
        #full grid of 308 points = 11 nodes x 28 cores/node
        xx,yy,zz = pr.set_up_grid(nx=8, ny=8, nz=7, xmin=-95, xmax=95, zmin=10, zmax=500, show_plot=False)

    position_list = list(zip(xx,yy,zz))

    if write_data:
        with open("position_list.txt",'w') as list_file:
            for pos in position_list:
                list_file.write(str(pos).replace(" ",""))
                list_file.write('\n')

    return np.array(position_list)

def get_gradB_magnitude(cached_B_tuple):
    """
    Calculates |B| and the magnitude of grad(|B|) from the cached
    tuple x, y, z, cached_bx, cached_by, cached_bz

    Uses spatial coordinates in orbit units x = {-100,100}
    """
    x, y, z, cached_bx, cached_by, cached_bz = cached_B_tuple
    dx = 100*np.abs(np.diff(x[:,0,0])[0])
    dy = 100*np.abs(np.diff(y[0,:,0])[0])
    dz = 100*np.abs(np.diff(z[0,0,:])[0])

    Bmagnitude = np.sqrt(cached_bx**2 + cached_by**2 + cached_bz**2)

    gradB = np.gradient(Bmagnitude, dx, dy, dz)

    gradB_magnitude = np.sqrt(np.sum(np.array(gradB)**2,axis=0))

    return gradB_magnitude

def get_gradB_scale_length(cached_B_tuple):
    """
    Returns |B|/|grad(|B|)|

    Uses spatial coordinates in orbit units x = {-100,100}
    """
    x, y, z, cached_bx, cached_by, cached_bz = cached_B_tuple
    dx = 100*np.abs(np.diff(x[:,0,0])[0])
    dy = 100*np.abs(np.diff(y[0,:,0])[0])
    dz = 100*np.abs(np.diff(z[0,0,:])[0])

    Bmagnitude = np.sqrt(cached_bx**2 + cached_by**2 + cached_bz**2)
    gradB = np.gradient(Bmagnitude, dx, dy, dz)
    gradB_magnitude = np.sqrt(np.sum(np.array(gradB)**2,axis=0))

    scale_length = Bmagnitude/gradB_magnitude

    return scale_length

def get_tensor_gradB(cached_B_tuple, abs=True):
    """
    Returns all nine components of $\nabla\vec{B}$ tensor.
    Uses spatial coordinates in orbit units x = {-100,100}
    """
    x, y, z, cached_bx, cached_by, cached_bz = cached_B_tuple
    dx = 100*np.abs(np.diff(x[:,0,0])[0])
    dy = 100*np.abs(np.diff(y[0,:,0])[0])
    dz = 100*np.abs(np.diff(z[0,0,:])[0])

    gradBx = np.gradient(cached_bx, dx, dy, dz)
    gradBy = np.gradient(cached_by, dx, dy, dz)
    gradBz = np.gradient(cached_bz, dx, dy, dz)

    if abs:
        return np.abs(np.array([gradBx, gradBy, gradBz]))
    else:
        return np.array([gradBx, gradBy, gradBz])


def get_tensor_scale_lengths(cached_B_tuple, grad_min=1e-4, grad_max=1e4, minimum_only=False, inside_cylinder=False):
    """
    Returns all nine components of the scale lengths defined by the
    \nabla\vec{B} tensor and the components of B.

    Uses spatial coordinates in orbit units x = {-100,100}
    """
    x, y, z, cached_bx, cached_by, cached_bz = cached_B_tuple
    dx = 100*np.abs(np.diff(x[:,0,0])[0])
    dy = 100*np.abs(np.diff(y[0,:,0])[0])
    dz = 100*np.abs(np.diff(z[0,0,:])[0])

    gradBx = np.abs(np.array(np.gradient(cached_bx, dx, dy, dz)))
    gradBy = np.abs(np.array(np.gradient(cached_by, dx, dy, dz)))
    gradBz = np.abs(np.array(np.gradient(cached_bz, dx, dy, dz)))

    LBx = cached_bx/np.clip(gradBx,grad_min,grad_max)
    LBy = cached_by/np.clip(gradBy,grad_min,grad_max)
    LBz = cached_bz/np.clip(gradBz,grad_min,grad_max)

    if minimum_only:
        return np.abs(np.array([LBx, LBy, LBz])).min(axis=0).min(axis=0)

    else:
        return np.abs(np.array([LBx, LBy, LBz]))


def get_vector_potential_from_B(x, y, z, Bx, By, Bz):
    """
    Uses force free condition to convert integral for A using J as a source
    to an integral using B as the source.
    ***SLOW BECAUSE THERE ARE O(N^2) OPERATIONS***
    """

    #since J is parallel to B, A is also parallel to B
    Bmag = np.sqrt(x**2 + By**2 + Bz**2)
    Amag = np.zeros_like(Bmag)

    #iterate over all r (field positions at which A is evaluated)
    for xindex_field, xfield in enumerate(x):
        print("Field x index (of 72 positions): ", xindex_field)
        for yindex_field, yfield in enumerate(y):
            for zindex_field, zfield in enumerate(z):
                rfield = np.array([xfield,yfield,zfield],dtype=float)

                #for each field position, integrate over all source positions
                #integral is B(r')dr'/|r-r'|
                #r' is the source location and r is the field location

                for xindex_source, xsource in enumerate(x):
                    start_time = time.time()
                    for yindex_source, ysource in enumerate(y):
                        for zindex_source, zsource in enumerate(z):
                            rsource = np.array([xsource,ysource,zsource], dtype=float)
                            script_r_vec = rfield - rsource #a la Griffiths
                            script_r = np.sqrt(script_r_vec**2).sum()
                            #component of integral
                            Amag[xindex_field, yindex_field, zindex_field] += Bmag[xindex_source, yindex_source, zindex_source]/script_r

                    print ("Full yz source point iteration: {}s".format(time.time() - start_time))

    return np.array(Amag,dtype=float).transpose([1,2,3,0])


def run_particle(initial_pos, initial_vel, field_data=None, b0=1.0, norbits=10, dt=0.01, dump_size=10000, data_dump_path=data_dump_path,write_data=True):
    """
    p1 = tt.run_particle([-65,-65,475],[0.09648947,-0.80964397,2.05985543],field_data=data)

    p1 = tt.run_particle([-55,55,425],[0.1,-0.8,2.0],field_data=data,norbits=5000,dt=0.1)

    """
    # loading the data, unless data is specified in calling the function
    if field_data is None:
        x, y, z, cached_bx, cached_by, cached_bz = load_taylor_field()
    else:
        x, y, z, cached_bx, cached_by, cached_bz = field_data

    taylor = Fields.interpolator(cached_bx, cached_by, cached_bz, x_shape=(-100, 100), y_shape=(-100, 100),
                                z_shape=(0, 1000), b0=b0)  # calling the interpolator

    p1 = pt.particle(initial_pos, initial_vel, dt, dump_size=dump_size,
                    data_dump_path=data_dump_path,write_data=write_data)
                    # creating the particle with the given conditions
    p1.set_boundaries(radius_max=100, z_max=1000)  # 10:1 Taylor state conditions
    p1.step(taylor.field, int(norbits / p1.dt)-1)  #subtract 1 to get nice # of iterations instead of nice # of orbits

    return p1

def read_full_trajectory(datafile, scale=None, nskip=1, vindex=None, return_all=False):
    '''
    Reads in h5 data produced by dumping particle data. H5 files will havee
    full phase space data as a function of time.

    Some one off position files are saved as csv with an array of shape (num_t,3).
    '''
    file_obj = h5py.File(datafile,mode='r')

    if vindex is None:
        r = file_obj['r'][()]
        v = file_obj['v'][()]
        B = file_obj['B'][()]
        dt = file_obj['dt'][()]
        iter = file_obj['iter'][()]
        outOfBounds = file_obj['outOfBounds'][()]

    else:
        r = file_obj[vindex]['r'][()]
        v = file_obj[vindex]['v'][()]
        B = file_obj[vindex]['B'][()]
        dt = file_obj[vindex]['dt'][()]
        iter = file_obj[vindex]['iter'][()]
        outOfBounds = file_obj[vindex]['outOfBounds'][()]


    print('Reading file: ', datafile, '\n')
    print('Parameters:')
    print('dt = ', dt)
    print('iter = ', iter)
    print('outOfBounds = ', outOfBounds)

    if scale=='single_orbits':
        r *= 100

    if return_all:
        return r[::nskip,:], v[::nskip,:], B[::nskip,:], dt, iter, outOfBounds
    else:
        return r[::nskip,:], v[::nskip,:], B[::nskip,:]

def get_B_along_traj(p1, field_data=None):
    '''
    :param p1: the particle for whose trajectory we want the B field
    :param field_data: in case we wish to use specific data
    gives us the B field at every position the particle inhabits during its motion
    '''
    N = p1.r.shape[0]
    t = np.arange(N) * p1.dt

    if field_data is None:
        x, y, z, cached_bx, cached_by, cached_bz = load_taylor_field()
    else:
        x, y, z, cached_bx, cached_by, cached_bz = field_data

    taylor_interp = pt.interpolator(cached_bx, cached_by, cached_bz,
                                    x_shape=(-100, 100), y_shape=(-100, 100),
                                    z_shape=(0, 1000), b0=1.0)
    B = []
    for ii in range(N):
        B.append(taylor_interp.field(p1.r[ii], t[ii]))  # getting the field for each position value

    return np.asarray(B)

def get_scalar_along_saved_traj(r_particle, scalar_field):
    '''
    :param r_particle: the r vector of the trajectory, shape(n_iter, 3)
    :param scalar_field: scalar evaluated at each field grid point (nx,ny,nz)
    '''
    N = r_particle.shape[0]

    scalar_interp = Fields.create_scalar_interpolator(scalar_field,
                                    x_shape=(-100, 100), y_shape=(-100, 100),
                                    z_shape=(0, 1000))

    return scalar_interp.scalar_field(r_particle)

def get_guiding_center(r,B,dt,get_width=False):
    """
    Calculate smoothed average position using minimum cyclotron frequency to set
    timescale.
    """
    Bmag = np.sqrt((B**2).sum(axis=-1))
    wc = Bmag.min() #q and m are 1 in our normalization
    averaging_time = 2/wc
    filter_width = averaging_time/dt
    if get_width:
        return filter_width
    r_gc = gaussian_filter1d(r,filter_width,axis=0)
    return r_gc

def get_cyl_coords(rvec,vvec):
    #
    # x = rvec[:,0]
    # y = rvec[:,1]
    # z = rvec[:,2]
    #
    # vx = vvec[:,0]
    # vy = vvec[:,1]
    # vz = vvec[:,2]
    #
    # azimuth = np.arctan(y/x)
    #
    # #rhat = np.cos(azimuth)*xhat

    r = np.sqrt(rvec[:,0]**2 + rvec[:,1]**2)
    vr_sign = np.sign(np.diff(r))
    vr = np.sqrt(vvec[1:,0]**2 + vvec[1:,1]**2)*vr_sign

    return r,vr

def get_vparallel(v,B):
    """
    Calculates the velocity component parallel to B
    at each point in the trajectory.

    :param v: velocity vector at each position in the trajectory
    :param B: field at each position in trajectory
    """

    #define unit vector in direction of B
    Bmagnitude = np.sqrt((B**2).sum(axis=1))
    bhat = (B.T/Bmagnitude).T #have to use transpose because shapes are set up backwards for broadcasting

    #get parallel component of velocity
    vparallel = v*bhat
    return vparallel

def get_vperp(v,B):
    """
    Calculates magnitude of velocity component perpendicular to B
    at each point in the trajectory.

    :param v: velocity vector at each position in the trajectory
    :param B: field at each position in trajectory
    """

    #define unit vector in direction of B
    Bmagnitude = np.sqrt((B**2).sum(axis=1))
    bhat = (B.T/Bmagnitude).T #have to use transpose because shapes are set up backwards for broadcasting

    #get parallel component of velocity
    vparallel = v*bhat

    #subtract vparallel from total v to get vperp
    vperp = v - vparallel

    #return only magnitude of vperp
    return np.sqrt((vperp**2).sum(axis=1))

def get_radial_component(r):
    return np.sqrt(r[:,0]**2 + r[:,1]**2)

def get_pitch_angle(v,B):
    """
    Takes arrays of v and B, calculates pitch angle element-wise.
    """
    vmag = np.sqrt((v*v).sum(axis=1))
    Bmag = np.sqrt((B*B).sum(axis=1))

    return np.arccos((v*B).sum(axis=1)/(vmag*Bmag))

def get_rL(v,B):
    """
    Calculates gyro (Larmor) radius using v_perp and B
    at each point in the trajectory.

    :param v: velocity vector at each position in the trajectory
    :param B: field at each position in trajectory
    """

    #define unit vector in direction of B
    Bmagnitude = np.sqrt((B**2).sum(axis=1))
    bhat = (B.T/Bmagnitude).T #have to use transpose because shapes are set up backwards for broadcasting

    #get parallel component of velocity
    vparallel = v*bhat

    #subtract vparallel from total v to get vperp
    vperp = v - vparallel

    #return r_L = m*v_perp / q*B
    return np.sqrt((vperp**2).sum(axis=1))/Bmagnitude

def get_mu_along_traj(v,B,m=1):
    """
    Calculates particle magnetic moment at each point in trajectory.

    :param p1: particle object containing trajectory info
    :param B: field at each position in trajectory
    :param m: mass of particle (default is normalized to 1)
    """

    vperp = get_vperp(v,B)
    Bmagnitude = np.sqrt((B**2).sum(axis=1))

    #return mu = m vperp**2 / 2B
    return m*vperp**2/(2*Bmagnitude)


def get_fluid_vorticity_along_r(r,v, medfilt=1):
    """
    Calculate particle vorticity using forward difference along trajectory.

    This also doesn't make a lot of sense, because fluid vorticity is about
    relative motion and the particle motion is only relative to itself.
    """
    # delta_vx = np.diff(v[:,0],axis=0)
    # delta_vy = np.diff(v[:,1],axis=0)
    # delta_vz = np.diff(v[:,2],axis=0)
    #
    # delta_x = np.diff(r[:,0],axis=0)
    # delta_y = np.diff(r[:,1],axis=0)
    # delta_z = np.diff(r[:,2],axis=0)

    #use second order differences
    delta_v = np.gradient(v,axis=0)
    delta_vx = delta_v[:,0]
    delta_vy = delta_v[:,1]
    delta_vz = delta_v[:,2]

    delta_r = np.gradient(r,axis=0)
    delta_x = delta_r[:,0]
    delta_y = delta_r[:,1]
    delta_z = delta_r[:,2]

    delta_r[np.abs(delta_r) < 1e-8] = 1e-8

    curl_v_x = delta_vz/delta_y - delta_vy/delta_z
    curl_v_y = delta_vz/delta_x - delta_vx/delta_z
    curl_v_z = delta_vy/delta_x - delta_vx/delta_y

    fluid_vorticity = np.array([curl_v_x,curl_v_y,curl_v_z]).transpose()
    return median_filter(fluid_vorticity, size=(medfilt,1))


def get_canonical_momentum_along_r(r,v,mesh_file):
    #mesh_file='cyl10_mesh.h5'
    ainterp_obj, binterp_obj = mktools.get_taylor_interpolators(mesh_file=mesh_file)
    #B = mktools.get_B_along_trajectory(r, binterp_obj)
    A = mktools.get_vector_potential_along_trajectory(r, ainterp_obj)

    return v + A #, B

def get_canonical_vorticity_along_r(r,v,B,medfilt=1):
    fluid_vorticity = get_fluid_vorticity_along_r(r,v,medfilt=medfilt)
    CV = fluid_vorticity + B
    return CV

def get_cv_components(r,cv,medfilt=10):
    """
    Extracts x,y,z; u,v,w for quiver plots
    """
    if medfilt is not None:
        cv_sm = median_filter(cv,size=(medfilt,1))
    else:
        cv_sm = cv
    x,y,z = np.squeeze(np.hsplit(r[1:,:],3))
    cvx, cvy, cvz = np.squeeze(np.hsplit(cv_sm,3))
    return x,y,z,cvx,cvy,cvz

def get_kinetic_helicity(velocity,vorticity):
    helicity = (velocity*vorticity).sum(axis=1)
    return helicity


def make_voxel_array(rvec, xres=10, yres=10, zres=100):
    """
    Transforms cloud of points from trajectory into array of ones and
    zeros to indicate the presence or absence of points in a voxel grid.
    """
    xvec = np.arange(-101,101,xres)
    yvec = np.arange(-101,101,yres)
    zvec = np.arange(0,1001,zres)

    voxel_x, voxel_y, voxel_z = np.meshgrid(xvec,yvec,zvec)

    for ii, xedge in enumerate(xvec):
        #make T/F array of positions where there are x trajectory values in
        # this bin of x-voxel indices
        xcondition = np.logical_and(rvec[:,0] > xedge,rvec[:,0] < xedge + 1)
        #if any elements are True, there is a trajectory point in that x voxel index, so look at the y coordinates

        #need to search y and z conditions for each bin in x with points


        # if xcondition.any():
        #     for jj, yedge in yvec:
        #         #look at all y locations where there is
        #         ycondition = np.logical_and(rvec[:,1] > yedge,rvec[:,1] < yedge + 1)
        #         if





def time_as_color_3D(rvec, ax=None, nskip=100, cmap=plt.cm.viridis, **kwargs):
    '''
    :param rvec: array of position vectors
    :param ax: set of axes, if needed to be specified beforehand
    :param cmap: just "colormap"
    :param kwargs: additional arguments
    makes a graph of the various position points in progressing colors
    '''
    x = rvec[::nskip, 0]
    y = rvec[::nskip, 1]
    z = rvec[::nskip, 2]

    if ax is None:
        ax = make_tube_axes()  # creates axes if not provided
    colors = np.linspace(0, 1, x.size, endpoint=True)
    ax.scatter(x, y, zs=z, c=cmap(colors), **kwargs)

    return ax


def color_by_scalar(rvec, scalar, ax=None, cmap=plt.cm.viridis,
                    colorbar=True,**kwargs):
    '''
    :param rvec: array of position vectors
    :param scalar: array of scalar variable to provide color (same length as rvec)
    :param ax: set of axes, if needed to be specified beforehand
    :param cmap: just "colormap"
    :param kwargs: additional arguments
    makes a graph of the various position points in progressing colors
    '''
    x = rvec[:, 0]
    y = rvec[:, 1]
    z = rvec[:, 2]

    if ax is None:
        ax = make_tube_axes()  # creates axes if not provided
    #scalar_range = scalar.max() - scalar.min()
    #colors = scalar/scalar_range #normalize range of scalar
    colors = scalar

    print(colors.shape)
    print(x.shape)


    ax.scatter(x, y, zs=z, c=colors, cmap=cmap, **kwargs)



def make_tube_axes(visit_scaling=False):
    '''
    :param visit_scaling: must be changed to true if the axes are already normalized/scaled properly
    This gives us a set of axes for our cylindrical Taylor state
    '''
    ax = plt.gcf().add_subplot(111, projection='3d')
    if visit_scaling:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, 10)
        set_axes_equal(ax)
        generate_cylinder((0, 0, 0), (0, 0, 10), 1, ax=plt.gca())
    else:
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_zlim(0, 1000)
        set_axes_equal(ax)
        generate_cylinder((0, 0, 0), (0, 0, 1000), 100, ax=plt.gca())
    return ax

def set_up_ax(title, ax):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)


def stringname_to_vector(vecstring):
    """
    Converts string representations of coordinates in filenames to
    vectors of numbers.
    """

    # find coordinate piece in between square brackets and remove underscores
    tempstring = vecstring.split('[')[1].split(']')[0].replace('_', '')

    # map each piece to the appropriate vector component (float) and return
    return np.array(tempstring.split('.')[0:3], float)


def merge_data_files(data_dir="/Users/adamdlight/SSX_data/particle_orbits/Taylor10t1/",
                     merged_name="merged_data.h5", silent=False):
    x_list = get_position_strings()

    # open file and then close it automatically when this block ends
    with h5py.File(data_dir + merged_name, mode='w') as joint_file:

        # Make a group because I was taught that you're not "supposed" to use the root group
        gg = joint_file.create_group('full_data_array')

        # Make a dataset to store the trajectories -- shape is (#gridpoints, #initial velocities, #coordinates, #timepoints)
        dd = gg.create_dataset('trajectories', (len(x_list), 100, 6, 601))
        dd.attrs['description'] = "Contains trajectory info vs time for all starting positions and velocities."
        dd.attrs['index_labels'] = "[num_positions,num_velocities,['x','y','z','vx','vy','vz'],'t']"
        iteration_info = gg.create_dataset('max_iterations',(len(x_list),100))
        iteration_info.attrs['description'] = "Contains maximum number of iterations completed before particle left volume."
        iteration_info.attrs['index_labels'] = "[num_positions,num_velocities]"

        # write data from each file to the combined file
        for pos_index, position_str in enumerate(x_list):
            if not silent:
                print(("Incorporating data from position " + position_str))
            filenames = get_vfile_list(position_str)

            for vel_index, eachfilename in enumerate(filenames):
                with h5py.File(eachfilename) as eachfile:
                    dd[pos_index,vel_index,0:3,:] = eachfile['r'].value.transpose()
                    dd[pos_index,vel_index,3:6,:] = eachfile['v'].value.transpose()
                    iteration_info[pos_index,vel_index] = eachfile['iter'].value

        print("Merged file created successfully.")






def get_trajectories_from_merged_file(data_dir="/Users/adamdlight/Dropbox/data/",
                                      merged_name="taylor_10t1_merged_data.h5",
                                      condition=None,time_index=0,
                                      return_full=False):
    """
    Returns particle trajectory data for particles that meet a boolean condition.

    If no condition is provided, the initial conditions of all trajectories
    that remain in the volume are returned.


    """

    if condition is None:
        #test case
        #returns all particles that stayed in the volume for the full simulation time
        with h5py.File(data_dir+merged_name, mode='r') as datafile:
            #read variable that contains the number of simulation iterations
            # during which each particle remained in the volume
            max_iterations = datafile['full_data_array']['max_iterations'].value

            #set condition to be that particle remains in volume for full simulation
            # (assumes at least one particle does so)
            condition = max_iterations == max_iterations.max()

            #save shapes of array so that we can recover the correct shape
            # for the trajectory array
            npositions,nvelocities = condition.shape
            ntrue = condition.sum()

            #reshape condition to match shape of trajectory initial conditions (npos,nvel,ncoords,ntimes)
            condition = np.broadcast_to(condition,(6,npositions,nvelocities)).transpose(1,2,0)

            #extract initial conditions ([...,0]) of entries where condition is met
            population = datafile['full_data_array']['trajectories'][...,time_index][condition]

            #reshape to be a 2D array with dimensions ntrue x ncoords
            population = population.reshape(ntrue,6)

            return population

    else:
        #returns all particle trajectories that match user provided condition
        #condition array should be the same dimensions as datafile['full_data_array']['max_iterations']
        with h5py.File(data_dir+merged_name, mode='r') as datafile:

            #save shapes of array so that we can recover the correct shape
            # for the trajectory array
            npositions,nvelocities = condition.shape
            ntrue = condition.sum()
            ntimes = datafile['full_data_array']['trajectories'].shape[-1]

            if return_full:
                #return x,v at all times
                # DON'T TRY THIS WITH A LARGE FRACTION OF THE PARTICLES
                # FOR EXAMPLE, I THINK THE MEMORY TAKEN UP WHEN LOADING
                # TRAJECTORIES FOR ALL CONFINED PARTICLES EXPANDS TO A
                # RIDICULOUS 60 GB.  I'M SURE THERE'S DUPLICATION.

                #reshape condition to match shape of trajectory initial conditions (npos,nvel,ncoords,ntimes)
                condition = np.broadcast_to(condition,(6,ntimes,npositions,nvelocities)).transpose(2,3,0,1)

                #extract only entries where condition is met
                population = datafile['full_data_array']['trajectories'][...,0:ntimes][condition]

                #reshape to be a 2D array with dimensions ntrue by ncoords by ntimes
                population = population.reshape(ntrue,6,ntimes)

            else:
                #reshape condition to match shape of trajectory initial conditions (npos,nvel,ncoords,ntimes)
                condition = np.broadcast_to(condition,(6,npositions,nvelocities)).transpose(1,2,0)

                #extract initial conditions ([...,0]) of entries where condition is met
                population = datafile['full_data_array']['trajectories'][...,0][condition]

                #reshape to be a 2D array with dimensions ntrue x ncoords
                population = population.reshape(ntrue,6)

            return population






def load_field_data(filename='cyl10_bfield_xyz.dat'):
    """
    Reads .dat text file like that from Chris Hansen.
    Parses (Nt,6) array into tuple of separate arrays (x,y,z,Bx,By,Bz)

    DEPRECATED BECAUSE IT DOESN'T RESHAPE ARRAYS AND DOESN'T
    PRESERVE GRID
    """

    data = np.loadtxt(filename, skiprows=3)
    # Return x,y,z,Bx,By,Bz
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4], data[:, 5]


def magnitude_histogram_vs_r(data=None, nbins=100, **kwargs):
    if data is None:
        x, y, z, Bx, By, Bz = load_field_data()

    else:
        x, y, z, Bx, By, Bz = data

    r = np.sqrt(x ** 2 + y ** 2)
    modB = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    # eliminate points outside the cylinder (each component is -1e99)
    cond = modB < 1e10
    r = r[cond]
    modB = modB[cond]

    H, yedges, xedges = np.histogram2d(r, modB, bins=nbins, normed=True)

    # multiply by total bin area to get PDF
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    H *= dx * dy * 100  # multiply by 100 to get percent

    xbins = xedges[:-1]  # +dx/2.
    ybins = yedges[:-1]  # +dy/2.

    # plotting the magnitude of the B field with respect to radius
    ax, cb, im = mytools.imview(H, x=xbins, y=ybins, **kwargs)

    im.set_clim(0, 0.1)
    ax.set_aspect('auto')
    cb.set_label('% of total grid locations', labelpad=8)
    ax.set_ylabel('Radius (norm.)')
    ax.set_xlabel('Field magnitude (arb.)')
    ax.set_title('|B| distribution in 10:1 Taylor state (CH 25 Apr, 2018)', y=1.02)  # may need to change date

    return ax, cb, im


def get_vfile_list(position_str):
    """
    Returns list of all files at particular initial position (all velocities)."
    """
    return glob.glob(trajectory_data_path + "taylor_volume[[]" + position_str[1:-1] + "[]]_*")


def get_xfile_list(velocity_str):
    """
    Returns list of all files at particular initial velocity (all positions)."
    """
    return glob.glob(trajectory_data_path + "taylor_volume*_[[]" + velocity_str[1:-1] + "[]].h5")


def get_position_strings():
    """
    Returns list of starting positions for Hari's summer 2018 run.

    Uses hard-coded set of initial velocities to extract set of positions,
    since there are only 100 velocities but over 6000 positions.
    """
    flist = get_xfile_list('[-0.02656608__1.15610407__0.308107__]')
    x_list = []

    prefix_length = len('taylor_volume')

    for ff in flist:
        # hard coded for my particular path
        #x_list.append(ff[81:97])
        start_ind = len(trajectory_data_path) + prefix_length
        x_list.append(ff[start_ind:start_ind+16])

    return x_list


def get_velocity_strings():
    vel_list = ['[-0.02656608__1.15610407__0.308107__]',
                '[-0.03122212__1.37717222__0.93326574]',
                '[-0.0444722__-0.34565906__1.90571449]',
                '[-0.08273727_-0.25617008__1.05333599]',
                '[-0.11320319_-1.51645793__1.06043518]',
                '[-0.12084555__0.66940807__0.56092951]',
                '[-0.15132834__0.69286088_-1.17846584]',
                '[-0.17857404__1.11899026_-0.60662871]',
                '[-0.19677797__0.38452203__0.21356223]',
                '[-0.2017833___1.76995124__1.22227115]',
                '[-0.30170841_-0.62143118_-1.30402747]',
                '[-0.31230124__0.37833885_-1.1346418_]',
                '[-0.33343165__0.63439578__1.07215169]',
                '[-0.4407115__-1.10895823__0.09818744]',
                '[-0.44244148__1.42928373__1.31361135]',
                '[-0.46139266_-1.41391314_-0.15778625]',
                '[-0.50472786__0.06921955__0.17223018]',
                '[-0.55647951_-1.2187859__-1.2560109_]',
                '[-0.68141215_-0.39014809__0.20154432]',
                '[-0.68234373_-0.90532681_-0.01164914]',
                '[-0.68276615_-0.8096551__-0.65817312]',
                '[-0.68349839__0.24398589__0.59142416]',
                '[-0.75531653__1.13668158_-2.15561776]',
                '[-0.80595625__0.72909775_-0.15065122]',
                '[-0.80623995__0.10454548__1.39511544]',
                '[-0.83972908_-0.18926516__0.20774838]',
                '[-0.86033689__0.70752077_-0.09972543]',
                '[-0.91954346__0.60504666_-0.0047967_]',
                '[-0.93495892__0.48265716_-0.43162303]',
                '[-0.93609284_-0.79097504__1.18542826]',
                '[-0.94367648__0.28691373__1.72293597]',
                '[-0.98730793_-1.12227931__2.30900987]',
                '[-1.01709563__0.51595169__1.20588553]',
                '[-1.02455292_-1.17102228_-0.69161279]',
                '[-1.09113381_-0.42386613__0.82512429]',
                '[-1.1004095___1.04635356__1.60785642]',
                '[-1.10952603_-0.06147662__1.70861783]',
                '[-1.15871564__0.8833391__-0.4140917_]',
                '[-1.16076199_-0.94880209_-0.02382131]',
                '[-1.24909327__0.06494328__0.33899312]',
                '[-1.32618107__0.07546287__0.5381046_]',
                '[-1.37616055_-0.46622161__0.75004382]',
                '[-1.39998712_-0.19072737__0.25199669]',
                '[-1.52982386__1.64946433_-1.70697719]',
                '[-1.59448752__0.41551564_-1.16930889]',
                '[-1.77436194_-1.63765043__0.02938844]',
                '[-1.82294793_-0.85937692__1.36889405]',
                '[-1.96919523_-1.71812594_-0.92908417]',
                '[-2.31344919__0.55558693_-0.75446147]',
                '[-2.74108471__0.96980307__1.9720353_]',
                '[0.09750471_0.02311336_1.47302669]',
                '[0.19608777_0.80546176_0.63761114]',
                '[0.21632307_0.89148019_0.70771796]',
                '[0.22808637_0.45330338_0.08707659]',
                '[0.37760597_0.35635874_1.46679851]',
                '[0.90578237_1.03818877_0.27266109]',
                '[0.92192425_0.28464265_0.95245741]',
                '[0.94828007_0.49634468_1.29011174]',
                '[1.03582888_0.28277042_0.01655404]',
                '[1.07538217_0.75261092_0.5736919_]',
                '[1.08390493_0.14493329_0.04987658]',
                '[1.21748975_0.80835128_0.22685571]',
                '[1.56731909_1.25779928_1.86594976]',
                '[1.762277___0.52356507_0.02950365]',
                '[2.49621142_1.4237695__0.54163674]',
                '[_0.02235136_-1.17934427__1.91670259]',
                '[_0.02561447_-1.36898497_-1.08943958]',
                '[_0.07848431__0.32574606_-1.5559996_]',
                '[_0.07992688_-0.18322591_-0.11444434]',
                '[_0.09173085_-0.50911842__1.45028048]',
                '[_0.09648947_-0.80964397__2.05985543]',
                '[_0.10242367_-0.54234314__1.80460493]',
                '[_0.14502719_-0.86959124_-0.36907235]',
                '[_0.16894461_-0.00743996_-1.97823589]',
                '[_0.19382009__0.14776432_-0.11972348]',
                '[_0.21364589_-2.06146701__0.72111834]',
                '[_0.24751284__0.2039392__-0.05861058]',
                '[_0.2542273__-0.06287609__1.45990116]',
                '[_0.32047994_-1.87186792_-0.8116722_]',
                '[_0.34616249_-0.97357446_-1.24803409]',
                '[_0.35505077__0.53659834_-0.20142186]',
                '[_0.35589736_-2.14025779_-0.05843929]',
                '[_0.37208371__0.429624___-0.96950881]',
                '[_0.41561677_-1.57582088_-2.23902375]',
                '[_0.47815227_-0.60563009_-0.92196904]',
                '[_0.50794728__0.10430782_-0.26594493]',
                '[_0.72031295_-0.28730619_-0.26890239]',
                '[_0.72989958_-2.36278339_-0.06349848]',
                '[_0.80542099_-2.45654704__0.30317497]',
                '[_0.83230215__0.3746059__-0.84517446]',
                '[_0.88222915_-1.00094824__0.42757002]',
                '[_0.90843515__2.08752617_-1.33617293]',
                '[_0.99063811__0.70217071_-0.66075613]',
                '[_1.03449497__1.19795459_-1.94183413]',
                '[_1.26781457_-1.37592088__0.62956229]',
                '[_1.48329552_-0.1577581___0.12953574]',
                '[_1.80896346__0.74198665_-2.83202594]',
                '[_1.84953666__0.56094746_-0.57119895]',
                '[_1.87049961_-0.9792867___0.02731044]',
                '[_2.07551398_-0.3589637__-0.31268221]']

    return vel_list


def check_file_list(traj_file_path='./data/trajectory_file_list_2018.txt'):
    """
    Check on actual trajectory files written during summer 2018.

    Goes through and matches filenames with the 100 velocities we think were simulated.
    """

    #get list of filenames derived from `ls -f taylor_volume | wc -l` on Bridges
    flist = np.loadtxt(traj_file_path,dtype=str).tolist()

    n_matched = []

    #for each velocity in our sample of 100
    velocity_list = get_velocity_strings()
    velocity_list.reverse()
    for vstring in velocity_list:
        #go through the whole file list
        num_matched = 0
        for ii,fname in enumerate(flist):
            #remove files from the file list that DO match each velocity
            if vstring in fname:
                flist.pop(ii)
                num_matched += 1
        print(("\nFor velocity "+vstring+":"))
        print(("the number of matching files is "+str(num_matched)))
        n_matched.append(num_matched)

    #return filenames that didn't match any of the 100 velocities
    return flist,n_matched  #remaining elements


# returns position dataset for a specific particle
def get_trajectory(filepath,silent=False):
    f = h5py.File(filepath, 'r');
    r = f['r'][()]
    x, y, z = (r[:, 0], r[:, 1], r[:, 2])
    if not silent:
        print("Simulated for %5d iterations." % f['iter'].value)
    f.close()
    return x, y, z

def get_trajectory_old(position_str, vel_str, silent=True):
    filepath = trajectory_data_path + "taylor_volume" + position_str + "_" + vel_str + ".h5"  # retrieving desired path
    f = h5py.File(filepath, 'r');
    r = f['r'].value
    x, y, z = (r[:, 0], r[:, 1], r[:, 2])
    if not silent:
        print("Simulated for %5d iterations." % f['iter'].value)
    f.close()
    return x, y, z


# possible particle orbits that look nice
# f = h5py.File(mypath+"taylor_volume[-65._-65._475.]_[_0.09648947_-0.80964397__2.05985543].h5",'r');  r = f['r'].value; v = f['v'].value; x,y,z = (r[:,0],r[:,1],r[:,2]); print f['iter'].value; f.close()

# f = h5py.File(mypath+"taylor_volume[-15.__75._425.]_[_0.09648947_-0.80964397__2.05985543].h5",'r');  r = f['r'].value; v = f['v'].value; x,y,z = (r[:,0],r[:,1],r[:,2]); print f['iter'].value; f.close(); ax.plot(x,y,z)

###f = h5py.File(mypath+"taylor_volume[-35.__25._275.]_[_0.09648947_-0.80964397__2.05985543].h5",'r');  r = f['r'].value; v = f['v'].value; x,y,z = (r[:,0],r[:,1],r[:,2]); print f['iter'].value; f.close(); ax.plot(x,y,z)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def generate_cylinder(start_point=[0,0,0], end_point=[0,0,1000], radius=100, ax=None, endcaps=False):
    """
    Hari's code to generate a cylindrical surface corresponding to the
    flux conserver.

    # Usual shape is:
    generate_cylinder((0,0,0),(0,0,1000),100)

    """
    # axis and radius
    p0 = np.array(start_point)  # point at one end
    p1 = np.array(end_point)  # point at other end
    R = radius

    # vector in direction of axis
    v = p1 - p0

    # find magnitude of vector
    mag = norm(v)

    # unit vector in direction of axis
    v = v / mag

    # make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    if (v == not_v).all():
        not_v = np.array([0, 1, 0])

    # make vector perpendicular to v
    n1 = np.cross(v, not_v)
    # normalize n1
    n1 /= norm(n1)

    # make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)

    # surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 2)
    theta = np.linspace(0, 2 * np.pi, 100)
    rsample = np.linspace(0, R, 2)

    # use meshgrid to make 2d arrays
    t, theta2 = np.meshgrid(t, theta)

    rsample, theta = np.meshgrid(rsample, theta)

    # generate coordinates for surface
    # "Tube"
    X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta2) * n1[i] + R * np.cos(theta2) * n2[i] for i in [0, 1, 2]]
    # "Bottom"
    X2, Y2, Z2 = [p0[i] + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    # "Top"
    X3, Y3, Z3 = [p0[i] + v[i] * mag + rsample[i] * np.sin(theta) * n1[i] + rsample[i] * np.cos(theta) * n2[i] for i in
                  [0, 1, 2]]

    if ax is None:
        ax = plt.gca()
    ax.plot_surface(X, Y, Z, color='blue', alpha=0.2)

    if endcaps:
        ax.plot_surface(X2, Y2, Z2, color='blue', alpha=0.2)
        ax.plot_surface(X3, Y3, Z3, color='blue', alpha=0.2)

    if ax.get_xlabel() == '':
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

"""
def get_simplified_shape(r,nplanes=10,plane_normal=2,
                         plane_thickness=0.01, plot=True,
                         ax=None,min_num_points=2,
                         plane_locations=None,
                         plot_points=False,
                         plot_polygons=True,
                         cmap=mytools.RB_linear()):
    
    Returns simplified vertex representation of puncture region
    for a specified number of planes normal to PLANE_NORMAL

    Calculates convex hull of points in each plane.
    
    if plane_locations is None:
        plane_locations = np.linspace(r[:,plane_normal].min(), r[:,plane_normal].max()-plane_thickness,nplanes,endpoint=True)
    else:
        nplanes = len(plane_locations)

    if plane_normal == 2:
        # planes are normal to the z axis

        hull_x = np.asarray([])
        hull_y = np.asarray([])
        hull_z = np.asarray([])

        for plane_index,zloc in enumerate(plane_locations):

            cond = mytools.in_limits(r[:,plane_normal], [zloc,zloc+plane_thickness],return_bool=True)

            if cond.sum() < min_num_points:
                #if there aren't enough points for a surface here, skip to the next plane
                print(("There are (only) %d points in this plane.  Skipping to next plane."%(cond.sum())))
                continue


            x1 = r[cond,0]
            x2 = r[cond,1]
            z = np.ones_like(x1)*zloc
            #return x1, x2

            hull = ConvexHull(np.asarray([x1,x2],float).transpose())

            hull_x = np.hstack([hull_x,x1[hull.vertices]])
            hull_y = np.hstack([hull_y,x2[hull.vertices]])
            hull_z = np.hstack([hull_z, z[hull.vertices]])
            #hull_y.append(x2[hull.vertices].tolist())
            #hull_z.append(z[hull.vertices])

            if plot:
                cindex = float(plane_index)/(nplanes-1)

                if ax is None:
                    ax = plt.gcf().add_subplot(111,projection='3d')

                if plot_points:
                    ax.plot(x1,x2,z,'o',alpha=1,color=cmap(cindex),markersize=6)

                if plot_polygons:
                    pp = plt.Polygon(np.asarray([x1[hull.vertices],x2[hull.vertices]]).transpose(),
                                    closed=True,alpha=0.2,facecolor=cmap(cindex),edgecolor='k')
                    ax.add_patch(pp)
                    art3d.pathpatch_2d_to_3d(pp, z=zloc, zdir="z")

                ax.set_xlim(-1,1)
                ax.set_ylim(-1,1)
                ax.set_zlim(0,10)
                plt.show()


        return hull_x, hull_y, hull_z

"""

def check_B_vs_dt(field_data=None, dts=[0.1,0.01,0.001], x0=[-55,55,425], v0=[0.1,-0.8,1.0],
                  norbits=10, dump_size=10000):

    if field_data is None:
        field_data = load_taylor_field()

    if float(norbits)/np.min(dts) > dump_size:
        dump_size += 10

    ax = plt.gcf().add_subplot(111)
    ax.set_xlabel("Time (fiducial orbits)")
    ax.set_ylabel("$B_x$")

    for dt in dts:

        p1 = run_particle(x0,v0,field_data=field_data,norbits=norbits,dt=dt,dump_size=dump_size)

        t = np.arange(p1.iter+1)*p1.dt
        ax.plot(t,p1.get_B()[:,0],label='dt = %1.4f'%dt)

    ax.legend().draggable()
    return p1
