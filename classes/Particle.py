import numpy as np
import time
from shapely.geometry import Polygon, Point

import Fields

class particle(object):

    def __init__(self, init_position=None, init_velocity=None, dt=None, nOfSteps=100, save_trajectories=False, check_turn=False):
        '''
        :param init_position:
        :param init_velocity:
        Removed the if statements because python started throwing an error since all doesn't operate on a bool like
        init_position != None
        '''
        self.r0 = init_position
        self.v0 = init_velocity
        self.m = 1
        self.q = 1
        self.dt = dt
        self.noOfSteps = nOfSteps
        self.iter = 0
        self.outOfBounds = False
        self.success = False
        
        self.save_trajectories = save_trajectories
        if self.save_trajectories:
            self.r = np.zeros((nOfSteps, 3))
            self.r[0] = init_position
            self.v = np.zeros((nOfSteps, 3))
            self.v[0] = init_velocity
            self.Bfield = np.zeros((nOfSteps, 3))
        else:
            self.r = init_position
            self.v = init_velocity
            self.check_turn = check_turn
            if self.check_turn:
                if self.v0[2] > 0:
                    self.positive_v0 = True
                elif self.v0[2] < 0:
                    self.positive_v0 = False

    def get_r(self):
        return self.r

    def get_v(self):
        return self.v

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
        
        # Checking initial point is within boundary
        if not self.bound.contains(Point(np.sqrt(self.r0[0]**2 + self.r0[1]**2), self.r0[2])):
            self.outOfBounds = True
            return self
        
        if self.save_trajectories:
            for i in range(0, self.noOfSteps-1):
                # check if particle is still within cube
                last_position = self.r[i]

                x = last_position[0]
                y = last_position[1]
                z = last_position[2]
                current_radius = np.sqrt(x ** 2 + y ** 2)

                if self.outOfBounds is True or not self.bound.contains(Point(current_radius, z)):
                    # set flag to true
                    self.outOfBounds = True

                    a = []
                    b = []
                    c = []

                    for j in range(0, (self.noOfSteps - self.iter)):
                        a.append([0, 0, 0])
                        b.append(last_position)
                        c.append(B(last_position,(self.iter+j)*self.dt))

                    self.v = np.concatenate([self.v, a],axis=0)
                    self.r = np.concatenate([self.r, b],axis=0)
                    self.Bfield = np.concatenate([self.Bfield, c],axis=0)

                    # quit the loop
                    break
                    
                t_help = q * self.dt * B(self.r[i], self.iter * self.dt) / (2 * m)
                v_prime = self.v[i] + np.cross(self.v[i], t_help)
                
                self.v[i+1] = self.v[i] + np.cross(v_prime, 2 * t_help / (1 + np.linalg.norm(t_help) ** 2))
                self.r[i+1] = self.r[i] + self.dt * self.v[i + 1]
                self.Bfield[i+1] = B(self.r[i], self.iter * self.dt)
                
                self.iter += 1
        
        elif not self.save_trajectories:
            for i in range(0, self.noOfSteps-1):
                # check if particle is still within cube
                last_position = self.r
                
                if self.outOfBounds or not self.bound.contains(Point(np.sqrt(last_position[0] ** 2 + last_position[1] ** 2), last_position[2])):
                    self.outOfBounds = True
                    if self.check_turn:
                        self.success = True
                    break
                
                t_help = q * self.dt * B(self.r, self.iter * self.dt) / (2 * m)
                v_prime = self.v + np.cross(self.v, t_help)
                
                self.v = self.v + np.cross(v_prime, 2 * t_help / (1 + np.linalg.norm(t_help) ** 2))
                self.r = self.r + self.dt * self.v
                
                if self.check_turn:
                    if self.positive_v0 and self.v[2] < 0:
                        self.iter = self.noOfSteps
                        self.success = True
                        break
                
                    elif not self.positive_v0 and self.v[2] > 0:
                        self.iter = self.noOfSteps
                        self.success = True
                        break
                
                self.iter += 1
        
        if not self.check_turn:
            self.success = True

        self.iter_time = (time.time() - start) / self.iter
            
        return self
