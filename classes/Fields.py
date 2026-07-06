import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy.core.multiarray import ndarray
from scipy.special import j0, j1, jn_zeros
import time
from scipy.interpolate import RegularGridInterpolator
import h5py

class interpolator(object):

    def __init__(self, bx, by, bz, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 1000), b0=1.0,
                 mask_max_radius=False):
        self.x_min = x_shape[0]
        self.x_max = x_shape[1]
        self.y_min = y_shape[0]
        self.y_max = y_shape[1]
        self.z_min = z_shape[0]
        self.z_max = z_shape[1]
        self.grid_shape = bx.shape  # we expect this to be the same as by.shape and bz.shape
        # might no longer need this, as fields are normalized at the generation stage
        self.b0 = b0
        self.mask_max_radius = mask_max_radius

        x = np.linspace(self.x_min, self.x_max, self.grid_shape[0])
        y = np.linspace(self.y_min, self.y_max, self.grid_shape[1])
        z = np.linspace(self.z_min, self.z_max, self.grid_shape[2])

        self.bx_function = RegularGridInterpolator((x, y, z), bx, method='linear',bounds_error=False)
        self.by_function = RegularGridInterpolator((x, y, z), by, method='linear',bounds_error=False)
        self.bz_function = RegularGridInterpolator((x, y, z), bz, method='linear',bounds_error=False)

    def field(self, r, t):
        if self.mask_max_radius is True:
            # print('Trigger 1')
            max_radius = (self.x_max - self.x_min) / 2
            # print(r[0] ** 2 + r[1] ** 2 , max_radius)
            if (r[0] ** 2 + r[1] ** 2 > max_radius ** 2):
                # print('Trigger 2')
                return np.asarray([0, 0, 0])

        bx = self.bx_function(r)[0]  # because RGI returns results in an array
        by = self.by_function(r)[0]
        bz = self.bz_function(r)[0]
        result = self.b0 * np.asarray([bx, by, bz])
        return result

class scalarInterpolator(object):

    def __init__(self, scalar, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 1000), normalize=False,
                 mask_max_radius=False):
        self.x_min = x_shape[0]
        self.x_max = x_shape[1]
        self.y_min = y_shape[0]
        self.y_max = y_shape[1]
        self.z_min = z_shape[0]
        self.z_max = z_shape[1]
        self.grid_shape = scalar.shape  # we expect this to be the same as by.shape and bz.shape
        # might no longer need this, as fields are normalized at the generation stage
        self.mask_max_radius = mask_max_radius

        x = np.linspace(self.x_min, self.x_max, self.grid_shape[0])
        y = np.linspace(self.y_min, self.y_max, self.grid_shape[1])
        z = np.linspace(self.z_min, self.z_max, self.grid_shape[2])

        if normalize:
            scalar_interp_data = scalar / np.abs(scalar).max()
        else:
            scalar_interp_data = scalar

        self.scalar_function = RegularGridInterpolator((x, y, z), scalar_interp_data, method='linear',bounds_error=False)

    def scalar_field(self, r):
        if self.mask_max_radius is True:
            # print('Trigger 1')
            max_radius = (self.x_max - self.x_min) / 2
            # print(r[0] ** 2 + r[1] ** 2 , max_radius)
            if (r[0] ** 2 + r[1] ** 2 > max_radius ** 2):
                # print('Trigger 2')
                return np.asarray([0, 0, 0])

        result = self.scalar_function(r)
        return result

def create_scalar_interpolator(scalar_field, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 1000),
                      normalize=False):
    return scalarInterpolator(scalar_field, x_shape=x_shape, y_shape=y_shape, z_shape=z_shape,
                          normalize=normalize)

def getUniformField(r, t):
    return np.asarray([1, 0, 0])



def nullField(r, t):
    return np.asarray([0, 0, 0])


def getSpheromakField(r, t):
    '''
    r =  array corresponding to the position at which we want the field
    t =  not sure - not used explicitly in the definition - perhaps meant to be for time evolution?
    returns the array of field components at various points
    '''
    def getSpheromakFieldAtPosition(x, y, z, center=(0, 0, 100), B0=1, R=100, L=100):
        '''
        The spheromak center in z must be L. Our spheromak has ratio L/R=1.
        '''

        # parameters
        j1_zero1 = jn_zeros(1, 1)[0]
        kr = j1_zero1 / R
        kz = np.pi / L

        lam = np.sqrt(kr ** 2 + kz ** 2)

        # construct cylindrical coordinates centered on center
        r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        theta = np.arctan2(y, x)
        centZ = z - center[2]

        # calculate cylindrical fields
        Br = -B0 * kz / kr * j1(kr * r) * np.cos(kz * centZ)
        Bt = B0 * lam / kr * j1(kr * r) * np.sin(kz * centZ)

        # convert back to cartesian, place on grid.
        Bx = Br * np.cos(theta) - Bt * np.sin(theta)
        By = Br * np.sin(theta) + Bt * np.cos(theta)
        Bz = B0 * j0(kr * r) * np.sin(kz * centZ)

        return Bx, By, Bz

    # using the previous function to determine the field components at every position observed
    Bx, By, Bz = getSpheromakFieldAtPosition(r[0], r[1], r[2])
    B_vector = np.asarray([Bx, By, Bz])

    return B_vector


def getDipoleField(r, t):
    # magnetic moment, s.t M0 = field strength one proton radius from origin on the xy plane
    M0 = -1000

    # for better readability
    x = r[0]
    y = r[1]
    z = r[2]

    # explicit calculation of the dipole field using position values
    B_vector = np.asarray([3 * M0 * x * z, 3 * M0 * y * z, M0 * (2 * z ** 2 - x ** 2 - y ** 2)])

    # normalizing the field?
    B_vector = B_vector / (np.dot(r, r) ** (5 / 2))

    return B_vector


def E(r, t):
    return np.asarray([0, 0, 0])




def create_spheromak_interpolator(grid_filename, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 100)):
    hf = h5py.File(grid_filename, 'r')
    bx_hf, by_hf, bz_hf = [hf.get(x) for x in ['bx', 'by', 'bz']]

    bx, by, bz = [np.zeros((x.shape)) for x in [bx_hf, by_hf, bz_hf]]

    for i in range(bx_hf.shape[0]):
        bx[i], by[i], bz[i] = [x[i] for x in [bx_hf, by_hf, bz_hf]]  # defining bx/by/bz based off of file values

    # The data masking gives us slightly inaccurate results near the edges
    '''
    for i in range(bx.shape[0]):
        for j in range(bx.shape[1]):
            for k in range(bx.shape[2]):
                # converting indices to coordinates

                x = x_shape[0] + i * (x_shape[1] - x_shape[0])/(bx.shape[0] - 1)
                y = y_shape[0] + j * (y_shape[1] - y_shape[0])/(by.shape[0] - 1)
                # z = z_shape[0] + k * (z_shape[1] - z_shape[0])/(bz.shape[0] - 1)

                # assuming that x_shape = y_shape
                max_radius = (x_shape[1] - x_shape[0]) / 2

                # masking
                if x ** 2 + y ** 2 > max_radius ** 2:
                    bx[i][j][k] = 0
                    by[i][j][k] = 0
                    bz[i][j][k] = 0
    '''

    hf.close()
    spheromak_interpolator = interpolator(bx, by, bz, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 100),
                                          mask_max_radius=True)
    return spheromak_interpolator


def create_taylor_interpolator(filename):
    sizes = (72, 72, 200)  # setting parameters for sizes (x, y, and z)
    data = np.loadtxt(filename, skiprows=3)  # loading the data from the file specified in calling the function

    '''
    x = data[:,0].reshape(sizes)
    y = data[:,1].reshape(sizes)
    z = data[:,2].reshape(sizes)
    '''
    # inputting the data in the respective positions (x data in first, y data in second, etc.)
    cached_bx = data[:, 3].reshape(sizes)  # same for magnetic field components
    cached_by = data[:, 4].reshape(sizes)
    cached_bz = data[:, 5].reshape(sizes)

    # setting values for the field components equal to 0 if they don't exceed a certain value
    for i in range(cached_bx.shape[0]):
        for j in range(cached_bx.shape[1]):
            for k in range(cached_bx.shape[2]):
                if (cached_bx[i][j][k] <= -1 * 10 ** 98):
                    cached_bx[i][j][k] = 0
                if (cached_by[i][j][k] <= -1 * 10 ** 98):
                    cached_by[i][j][k] = 0
                if (cached_bz[i][j][k] <= -1 * 10 ** 98):
                    cached_bz[i][j][k] = 0

    # final interpolation using the values from the file after adjustment, which is then returned
    taylor = interpolator(cached_bx, cached_by, cached_bz, x_shape=(-100, 100), y_shape=(-100, 100), z_shape=(0, 1000),
                          b0=1 / 0.38239010712927873)

    return taylor
