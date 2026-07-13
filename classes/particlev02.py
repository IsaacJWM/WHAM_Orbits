import numpy as np
import sys
import time
import h5py
from shapely.geometry import Polygon, Point

import Fields

class particle(object):

    def __init__(self, init_position=None, init_velocity=None, dt=None, nOfSteps=100, dump_size=10000,data_dump_path='./',write_data=True,silent=False):
        '''
        :param init_position:
        :param init_velocity:
        Removed the if statements because python started throwing an error since all doesn't operate on a bool like
        init_position != None
        '''
        self.r0 = init_position
        self.r = np.zeros((nOfSteps, 3))
        self.r[0] = init_position
        self.v0 = init_velocity
        self.v = np.zeros((nOfSteps, 3))
        self.v[0] = init_velocity
        self.m = 1
        self.q = 1
        self.dt = dt
        self.noOfSteps = nOfSteps
        # better to count the number of iterations as opposed to the time
        self.iter = 0
        self.outOfBounds = False
        self.line_step = 0.01
        self.dump_size = dump_size
        self.dump_path = data_dump_path
        self.write_data = write_data
        self.Bfield = np.zeros((nOfSteps, 3))
        self.silent = silent

    def get_r(self):
        return self.r

    def get_v(self):
        return self.v

    def get_B(self):
        return self.Bfield

    def set_boundaries(self, vertices):
        '''
        :param vertices: the vertices of the shape in the r-z plane defining
        the boundary of confined particles.
        Used to set the boundary of our container
        '''
        self.bound = Polygon(vertices)

    def step(self, B, E=Fields.nullField):
        '''
        Returns the particle object itself, after having Boris Pushed it through noOfSteps steps, in a magnetic field B.
        Inputs:
        B(r, t): function that returns the magnetic field vector
        noOfSteps: number of iterations you want to push the particle
        '''

        start = time.time()

        m = self.m
        q = self.q

        self.Bfield[0] = B(self.r0, self.iter * self.dt)

        ####### debug
        #self.dump_size=100
        #######
        if self.write_data:
            dump_size = self.dump_size
        else:
            dump_size = self.noOfSteps+1

        # only using the self notation for parameters which need to be updated
        # such as v, r, and t. In other words, all parameters on the left
        # hand side that we intend to preserve

        # if many, many steps are required, want to dump data in chunks and not
        # keep it all in memory
        if self.noOfSteps > dump_size:
            #reduce number of steps to be an integer number of data dumps
            nBlocks = int(np.floor(self.noOfSteps/dump_size))
            if not self.silent:
                print("Total number of steps will be {:d} blocks by {:d} iterations = {:d}".format(nBlocks+1,dump_size,(nBlocks+1)*dump_size))

            #set number of steps for each block to be dumpsize
            self.noOfSteps = dump_size

        else:
            nBlocks = 0

        for block in range(nBlocks+1):
            if self.outOfBounds == True:
                break

            #include initial position in number of steps so that total length = num_iterations
            for ii in range(0, self.noOfSteps-1):
                # check if particle is still within cube
                last_position = self.r[ii]

                if ii == 100 and not self.silent:
                    print("Estimated total iteration time for this particle: {:4.1f} s".format(self.noOfSteps*(time.time() - start) / self.iter))

                x = last_position[0]
                y = last_position[1]
                z = last_position[2]
                current_radius = np.sqrt(x ** 2 + y ** 2)

                if self.outOfBounds is True or not self.bound.contains(Point(current_radius, z)):
                    # set flag to true
                    self.outOfBounds = True
                    self.write_data = False

                    # pad the rest of the array with the previous position
                    '''
                    self.v.extend((iter_count - self.iter) * [0, 0, 0])
                    self.r.extend((iter_count - self.iter) * last_position)
                    '''
                    a = []
                    b = []
                    c = []

                    for j in range(0, (self.noOfSteps*(block+1) - self.iter)):
                        a.append([0, 0, 0])
                        b.append(last_position)
                        c.append(B(last_position,(self.iter+j)*self.dt))

                    self.v = np.concatenate([self.v, a],axis=0)
                    self.r = np.concatenate([self.r, b],axis=0)
                    self.Bfield = np.concatenate([self.Bfield, c],axis=0)

                    # quit the loop
                    break

                #half_electrical_impulse = q * self.dt * E(self.r[ii], self.iter * self.dt) / (2 * m)
                #v_minus = self.v[ii] + half_electrical_impulse

                #t_help = q * self.dt * B(self.r[ii], self.iter * self.dt) / (2 * m)
                #v_prime = v_minus + np.cross(v_minus, t_help)

                #s_help = 2 * t_help / (1 + np.linalg.norm(t_help) ** 2)
                #v_plus = v_minus + np.cross(v_prime, s_help)

                #self.v[ii+1] = v_plus + half_electrical_impulse
                
                t_help = q * self.dt * B(self.r[ii], self.iter * self.dt) / (2 * m)
                v_prime = self.v[ii] + np.cross(self.v[ii], t_help)
                
                self.v[ii+1] = self.v[ii] + np.cross(v_prime, 2 * t_help / (1 + np.linalg.norm(t_help) ** 2))
                self.r[ii+1] = self.r[ii] + self.dt * self.v[ii + 1]
                self.Bfield[ii+1] = B(self.r[ii], self.iter * self.dt)

                self.iter += 1
                
            # now save these time points to disk and set up new v and r arrays
            # with the inital value being the value at the current iteration to save memory

            
            self.iter_time = (time.time() - start) / self.iter

            if self.write_data:
                if not self.silent:
                    print("\nDumping data...")
                self.dump_data()

            if not self.silent:
                print("Time taken to execute " + str(self.iter) + " iterations: " + str(time.time() - start)+' s')
                print("Average iteration time: {:f} s".format(self.iter_time))


        return self


    def dump_data(self,nsteps=0,
                  overwrite=False):
        # creates an h5 file with information about the various parameters of the particle
        initial_velocity = self.v0
        initial_position = self.r0

        filename = self.dump_path + 'traj{:d}_'.format(nsteps) + str(initial_position) + "_" + str(initial_velocity) + ".h5"
        # getting rid of spaces to make the files easy to load
        filename = filename.replace(' ', '_')

        if self.iter < self.dump_size:
            only_one_dump = True
            dump_size = self.iter
        else:
            only_one_dump = False
            dump_size = self.dump_size

        #check if file exists for this trajectory
        #if not, initialize and write first dump

        if self.iter == dump_size:
            #this is the first time the data is dumped
            if overwrite or only_one_dump:
                write_mode='w'
            else:
                write_mode='w-'

            try:
                with h5py.File(filename,write_mode) as hf:

                    # subtract 1 from the shape for initial condition
                    # leave off last value as it will be included as initial value
                    # in the next block
                    write_length = np.asarray(self.r).shape[0]
                    rdata = hf.create_dataset("r", (write_length-1,3), maxshape=(None, 3))
                    rdata[:,:] = self.r[:-1]

                    vdata = hf.create_dataset("v", (write_length-1,3), maxshape=(None, 3))
                    vdata[:,:] = self.v[:-1]

                    Bdata = hf.create_dataset("B", (write_length-1,3), maxshape=(None, 3))
                    Bdata[:,:] = self.Bfield[:-1]

                    hf.create_dataset('iter', data=[self.iter])
                    hf.create_dataset('outOfBounds', data=[self.outOfBounds])
                    hf.create_dataset('dt', data=[self.dt])

            except IOError as fileerr:
                # for debugging
                print(fileerr)

                #if the file already exists, ask if you want to overwrite
                print("\nThis data file already exists.")
                user_input = input("Do you wish to overwrite it? (y/N)")

                if user_input == 'y':
                    #self.dump_data(nsteps=nsteps,location=location,overwrite=True)
                    self.dump_data(nsteps=nsteps,overwrite=True)
                    print("Data file will be overwritten.  Continuing with iterations...")
                else:
                    sys.exit("Canceling data dump.\n")
                    pass

        #otherwise append to existing datasets
        else:
            with h5py.File(filename,'a') as hf:
                rdata = hf["r"]
                vdata = hf["v"]
                Bdata = hf["B"]

                #debugging
                #print(rdata.shape)
                #print np.asarray(self.r).shape

                #append to existing data
                rdata.resize((rdata.shape[0] + dump_size, 3))
                rdata[-dump_size:] = self.r[:-1]

                vdata.resize((vdata.shape[0] + dump_size, 3))
                vdata[-dump_size:] = self.v[:-1]

                Bdata.resize((Bdata.shape[0] + dump_size, 3))
                Bdata[-dump_size:] = self.Bfield[:-1]

                hf["iter"][...] = [rdata.shape[0]]
                hf["outOfBounds"][...] = [self.outOfBounds]

        if not only_one_dump:
            self.v = [self.v[-1]]
            self.r = [self.r[-1]]
            self.Bfield = [self.Bfield[-1]]
        return self

    def get_vperp(self):
        """
        Calculates magnitude of velocity component perpendicular to B
        at each point in the trajectory.

        """

        #define unit vector in direction of B
        Bmagnitude = np.sqrt((self.get_Bfield()**2).sum(axis=1))
        bhat = (self.get_Bfield().T/Bmagnitude).T #have to use transpose because shapes are set up backwards for broadcasting

        #get parallel component of velocity
        vparallel = self.get_v()*bhat

        #subtract vparallel from total v to get vperp
        vperp = self.get_v() - vparallel

        #return only magnitude of vperp
        return np.sqrt((vperp**2).sum(axis=1))

    def get_mu_along_traj(self, m=1):
        """
        Calculates particle magnetic moment at each point in trajectory.
        :param m: mass of particle (default is normalized to m_p = 1)
        """

        vperp = self.get_vperp()
        Bmagnitude = np.sqrt((self.get_Bfield**2).sum(axis=1))

        #return mu = m vperp**2 / 2B
        return m*vperp**2/(2*Bmagnitude)
