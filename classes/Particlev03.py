import numpy as np
import time
from shapely.geometry import Polygon, Point

import Fields

class particle(object):

    def __init__(self, init_position=None, init_velocity=None, dt=None, nOfSteps=100, silent=False):
        '''
        :param init_position:
        :param init_velocity:
        Removed the if statements because python started throwing an error since all doesn't operate on a bool like
        init_position != None
        '''
        self.r0 = init_position
        self.r = init_position
        self.v0 = init_velocity
        self.v = init_velocity
        self.m = 1
        self.q = 1
        self.dt = dt
        self.noOfSteps = nOfSteps
        # better to count the number of iterations as opposed to the time
        self.iter = 0
        self.outOfBounds = False
        self.silent = silent
        self.success = False
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
        
        for i in range(0, self.noOfSteps-1):
            # check if particle is still within cube
            last_position = self.r
            
            if self.outOfBounds is True or not self.bound.contains(Point(np.sqrt(last_position[0] ** 2 + last_position[1] ** 2), last_position[2])):
                self.outOfBounds = True
                break
            
            t_help = q * self.dt * B(self.r, self.iter * self.dt) / (2 * m)
            v_prime = self.v + np.cross(self.v, t_help)
            
            self.v = self.v + np.cross(v_prime, 2 * t_help / (1 + np.linalg.norm(t_help) ** 2))
            self.r = self.r + self.dt * self.v
            
            if self.positive_v0 and self.v[2] < 0:
                self.iter = self.noOfSteps
                self.success = True
                break
            
            elif not self.positive_v0 and self.v[2] > 0:
                self.iter = self.noOfSteps
                self.success = True
                break
            
            self.iter += 1
                
        self.iter_time = (time.time() - start) / self.iter
            
        return self
