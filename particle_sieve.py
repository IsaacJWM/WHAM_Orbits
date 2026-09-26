import numpy as np
import h5py
import sys

def select_velocities(nvel=100, rng=None):
    if rng is None:
        rng = np.random.default_rng()  # unseeded, uses OS entropy
    return rng.standard_normal(nvel)


def write_single_position_data(p1,filename,groupname,write_mode='w-'):

    try:
        with h5py.File(filename,write_mode) as hf:
            #create a new group if it doesn't already exist
            while groupname in hf.keys():
                print("Duplicate group name: "+groupname)
                groupname = groupname + "d"
            gp = hf.create_group(groupname)

            gp.create_dataset('r',data=p1.r[:-1])
            gp.create_dataset('v',data=p1.v[:-1])
            # subtract 1 from the shape for initial condition

            gp.create_dataset('iter', data=[p1.iter])
            gp.create_dataset('outOfBounds', data=[p1.outOfBounds])
            gp.create_dataset('dt', data=[p1.dt])
            
            hf.close()

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
