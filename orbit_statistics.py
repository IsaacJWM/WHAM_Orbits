import numpy as np
import h5py
import sys
import os
import time
import matplotlib
import matplotlib.pyplot as plt
import argparse

import mytools
import taylor_field_tools as tft
import particle_sieve as pr

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from collections import Counter
from pathlib import Path
home = Path.home()
data_loc = os.path.join(home,"Dropbox","python","particle_orbits","data")

class particle_summary(dict):

    def __init__(self,r0=None,v0=None,iterations=None,confined=None):
        """
        Reads file into set of arrays?
        """
        pass


def read_single_position_files(directory,read_escaped=False,
                               save_output=True, outfile=None):
    """
    Collect initial conditions, residence time,
    """

    initial_positions = []
    final_positions = []
    initial_velocities = []
    initial_fields = []
    max_iterations = []
    is_confined = []

    file_list = os.listdir(directory)

    if read_escaped:
        file_ending = 'escaped.h5'
    else:
        file_ending = 'confined.h5'

    for file_name in file_list:

        #if it's not the kind particle we want, skip to the next file
        if not file_name.endswith(file_ending):
            continue

        #otherwise, read in the data
        file_path = os.path.join(directory,file_name)

        with h5py.File(file_path,'r') as ff:

            for gp in ff.keys():

                initial_positions.append(ff[gp]['r'][0])
                final_positions.append(ff[gp]['r'][-1])
                initial_velocities.append(ff[gp]['v'][0])
                initial_fields.append(ff[gp]['B'][0])
                max_iterations.append(ff[gp]['iter'][0])
                is_confined.append(not(ff[gp]['outOfBounds'][0]))

    if save_output:
        if outfile is None:
            dir_name = directory.lstrip('.').lstrip('/').rstrip('/')
            outfile = "particle_summary_"+dir_name+"_"+file_ending[:-3]+".npz"
            print("Writing to file "+outfile+'\n')

        np.savez(outfile,initial_positions=np.asarray(initial_positions),
                final_positions=np.asarray(final_positions),
                initial_velocities=np.asarray(initial_velocities),
                initial_fields=np.asarray(initial_fields),
                max_iterations=np.asarray(max_iterations),
                is_confined=np.asarray(is_confined))

    return np.asarray(initial_positions),np.asarray(final_positions),  np.asarray(initial_velocities), np.asarray(initial_fields), np.asarray(max_iterations), np.asarray(is_confined)
#       read r0, v0, iterations, confined into giant array
#


def read_single_trajectory_from_merged_file(filepath, vel_index):
    with h5py.File(filepath, 'r') as ff:
        r = ff[vel_index]['r'][()]
        v = ff[vel_index]['v'][()]
        B = ff[vel_index]['B'][()]
        max_iterations = (ff[vel_index]['iter'][0])
        is_confined = (not(ff[vel_index]['outOfBounds'][0]))

    print("Initial position: {}\n Initial velocity: {}".format(r[0,:],v[0,:]))

    return r, v, B, max_iterations, is_confined

def read_single_h5_trajectory(filepath):
    with h5py.File(filepath, 'r') as ff:
        r = ff['r'][()]
        v = ff['v'][()]
        B = ff['B'][()]
        max_iterations = (ff['iter'][0])
        is_confined = (not(ff['outOfBounds'][0]))

    print("Initial position: {}\n Initial velocity: {}".format(r[0,:],v[0,:]))

    return r, v, B, max_iterations, is_confined


def read_summary_data(filename, legacy=False):
    """
    Reads summary data from .npz file created by read_single_position_files()

    LEGACY files do not contain final position data, so that has to be left out
    when reading
    """

    data = np.load(filename)

    ri = data['initial_positions']
    if legacy:
        rf = None
    else:
        rf = data['final_positions']
    vi = data['initial_velocities']
    Bi = data['initial_fields']
    max_iterations = data['max_iterations']
    is_confined = data['is_confined']

    return ri, rf, vi, Bi, max_iterations, is_confined

def combine_esc_conf_summaries(esc_file,conf_file,legacy=False):
    ri_esc, rf_esc, vi_esc, Bi_esc, max_iterations_esc, not_confined = read_summary_data(esc_file,legacy=legacy)
    ri_conf, rf_conf, vi_conf, Bi_conf, max_iterations_conf, is_confined = read_summary_data(conf_file,legacy=legacy)

    ri = np.concatenate([ri_esc,ri_conf])
    if legacy:
        rf = None
    else:
        rf = np.concatenate([rf_esc,rf_conf])
    vi = np.concatenate([vi_esc,vi_conf])
    Bi = np.concatenate([Bi_esc,Bi_conf])
    max_iterations = np.concatenate([max_iterations_esc,max_iterations_conf])
    is_confined = np.concatenate([not_confined,is_confined])

    return ri, rf, vi, Bi, max_iterations, is_confined


def combine_subruns(file_list,legacy=False,write_new_summary=True,savefile="concatenated_summaries.npz"):
    """
    merge_rect_file_list = ["../particle_summary_trajectory_data_6524539_escaped.npz", "../particle_summary_trajectory_merge_6524539_7294312_escaped.npz", "../particle_summary_merge_6524539_7294312_10186475_escaped.npz", "../particle_summary_merge_6524539_7294312_10186475_confined.npz"]
    """

    ri, rf, vi, Bi, max_iterations, is_confined = collect_subrun_stats(file_list,[0,0,0,0],legacy=True)

    if write_new_summary:
        np.savez(savefile,initial_positions=np.asarray(ri),
                final_positions=np.asarray(rf),
                initial_velocities=np.asarray(vi),
                initial_fields=np.asarray(Bi),
                max_iterations=np.asarray(max_iterations),
                is_confined=np.asarray(is_confined))

    return ri, rf, vi, Bi, max_iterations, is_confined


# def get_confined_trajectory_list(data_dir):
#     """
#     Returns file names and initial velocities for trajectories that
#     are confined through end of run.
#
#     file_list = get_confined_file_list('data/trajectory_data_10186475')
#     """
#     ri, rf, vi, Bi, max_iterations, is_confined = read_summary_data(summary_data_file)
#
#     confined_indices = np.where(is_confined)
#     file_list = []
#     vi_list = []
#
#     for ind in confined_indices:
#         file_list.append("trajectories_x{+d}y{+d}z{+d}".format(ri[ind,0],ri[ind,1],ri[ind,2]))
#         vi_list.append(vi[ind,:])
#
#     return file_list, vi_list


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
                ax.set_xlim(-100,100)
                ax.set_ylim(-100,100)
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


def get_params_from_image_directory(image_directory):

	file_list = os.listdir(image_directory)
	r0 = []
	v0 = []
	B0 = []

	for image_filename in file_list:
		if image_filename[:12] != 'trajectories':
			continue
		rimage, vimage, Bimage =  get_params_from_image_filename(image_filename)
		r0.append(rimage)
		v0.append(vimage)
		B0.append(Bimage)
	return np.asarray(r0), np.asarray(v0), np.asarray(B0)


def get_params_from_image_filename(image_filename, data_dir = '/Volumes/alight/trajectory_merge_6524539_7294312_10186475/'):
	velocity_string = image_filename[-8:-4]
	data_filename = image_filename[:-9]+'.h5'

	with h5py.File(os.path.join(data_dir,data_filename), mode='r') as datafile:
		r0 = datafile[velocity_string]['r'][0]
		v0 = datafile[velocity_string]['v'][0]
		B0 = datafile[velocity_string]['B'][0]
	return r0, v0, B0


def collect_iter_counts(file_list,n_previous_iter):
    """
    Collects max_iter counts for particles in a sequence of continued runs.

    FILE_LIST should be a list of data summary files in time order, including only
    escaped particles except for the final run.

    N_PREVIOUS_ITER is a list of iterations completed in the run before each
    of the runs in the summary files.  The first element should always be
    zero (beginning of full concantenated run).  If each run is e.g. 3000 orbits at dt=0.01, then the array would read something like [0,300000,300000,300000].

    """
    #collect all escaped particles from each run
    for ii,data_file in enumerate(file_list):
        iter_counts = read_summary_data(data_file)[4] + n_previous_iter[ii]
        if ii==0:
            all_particle_iters = iter_counts
        else:
            all_particle_iters = np.concatenate([all_particle_iters,iter_counts])

    return all_particle_iters


def collect_subrun_stats(file_list,n_previous_iter,legacy=False):
    """
    Collects all particle summary stats for particles in a sequence of continued runs.

    FILE_LIST should be a list of data summary files in time order.

    N_PREVIOUS_ITER is a list of iterations completed before each
    of the runs in the summary files.  The first element should always be
    zero (beginning of full concantenated run).  If each run is e.g. 3000 orbits at dt=0.01, then the array would read something like [0,300000,600000,900000].


In [340]: file_list = [os.path.join(datadir,'particle_summary_trajectory_data_65
     ...: 24539_escaped.npz'), os.path.join(datadir,'particle_summary_trajectory
     ...: _data_7294312_escaped.npz'),os.path.join(datadir,'particle_summary_tra
     ...: jectory_data_10186475_escaped.npz')]

In [340]: ri, rf, vi, Bi, max_iterations, is_confined = ostats.collect_subrun_st
     ...: ats(file_list,[0,300000,600000],legacy=True)



     ******CAUTION********
     if you use the above tactic, then you will get different initial positions for each subrun and you can't collect stats by starting position at time 0

     WORKAROUND: USE MERGED FILES

     In [514]: file_list = [os.path.join(datadir,'particle_summary_trajectory_data_65
     ...: 24539_escaped.npz'), os.path.join(datadir,'particle_summary_merge_6524
     ...: 539_7294312_escaped.npz'),os.path.join(datadir,'particle_summary_merge
     ...: _6524539_7294312_10186475_escaped.npz')]

     In [515]: ri, rf, vi, Bi, max_iterations, is_confined = ostats.collect_subrun_st
     ...: ats(file_list,[0,0,0],legacy=True)

    """

    #collect all escaped particles from each run
    for ii,data_file in enumerate(file_list):
        if ii==0:
            initial_positions, final_positions, initial_velocities, initial_fields, max_iterations, is_confined = read_summary_data(data_file,legacy=legacy)
            print('File number 0')
            print(initial_positions.shape)
            print()

        else:
            r0_temp, rf_temp, v0_temp, B0_temp, max_iter_temp, is_confined_temp = read_summary_data(data_file,legacy=legacy)

            initial_positions = np.concatenate([initial_positions,r0_temp])

            print('File number {}'.format(ii))
            print(r0_temp.shape)
            print(initial_positions.shape)
            print()

            if not legacy:
                final_positions = np.concatenate([final_positions,rf_temp])

            initial_velocities = np.concatenate([initial_velocities,v0_temp])
            initial_fields = np.concatenate([initial_fields,B0_temp])

            iter_counts = max_iter_temp + n_previous_iter[ii]
            max_iterations = np.concatenate([max_iterations,iter_counts])

            is_confined = np.concatenate([is_confined,is_confined_temp])


    return initial_positions, final_positions, initial_velocities, initial_fields, max_iterations, is_confined


def confinement_vs_time(max_iterations,ntot=30800,total_iter=300000,
                        return_number=False):
    """
    Calculates the fraction of particles still confined after certain
    number of iterations using the escaped particle summary information.

    ntot = 308 positions x 100 velocities
    total_iter = 3000 orbits x 100 time points per orbit

    Input is a list/array of all particles iteration count; if
    there is more than one run continuing the trajectories,
    then it's necessary to build a list of counts from the different
    directories by reading in the summary data for all escaped particles
    in the un-merged directories and the
    confined particles in the final merged directory.
    (See collect_iter_counts)
    """

    N_escaped = []

    for iteration in range(total_iter):
        already_escaped = np.where(max_iterations < iteration)[0].shape
        N_escaped.append(already_escaped)

    nconfined = ntot - np.asarray(N_escaped)

    if return_number:
        return N_escaped

    return nconfined/ntot


def plot_confinement_fraction_vs_time(fraction,dt=0.01,loglog=False,plot_loss=False,exclude_before=24366,**kwargs):

    ax = plt.gcf().add_subplot(111)

    norbits = np.linspace(0,fraction.size*dt,num=fraction.size)

    start_index = exclude_before
    xfit = np.log10(norbits[start_index:])
    yfit = fraction[start_index:]
    fit = np.polyfit(xfit,yfit,1)

    if loglog:
        ax.loglog(norbits,fraction,**kwargs)
        ax.set_xlim(1,norbits.max())
        ax.set_ylim(1e-1,1)
        fmt = '%.1f' # Format you want the ticks, e.g. '40%'
        yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        ax.yaxis.set_minor_formatter(yticks)
        ax.set_xlabel('Number of Fiducial Orbits')
        ax.set_ylabel('Fraction of Initial Particles Confined')
        ax.set_title('Particle Confinement Fraction During Simulation',y=1.01)
    elif plot_loss:
        ax.semilogx(norbits,1-fraction,**kwargs)
        ax.set_xlim(1,norbits.max())
        ax.set_ylim(1e-2,1)
        fmt = '%.1f' # Format you want the ticks, e.g. '40%'
        yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        ax.yaxis.set_minor_formatter(yticks)
        ax.set_xlabel('Number of Fiducial Orbits')
        ax.set_ylabel('Fraction of Initial Particles Lost')
        ax.set_title('Particle Loss Fraction During Simulation',y=1.01)
    else:
        ax.plot(norbits,fraction*100,**kwargs)
        ax.set_xlim(0,norbits.max())
        ax.set_ylim(0,100)
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        ax.set_xlabel('Number of Fiducial Orbits')
        ax.set_ylabel('Percentage of Initial Particles Confined')
        ax.grid()
        ax.set_title('Particle Confinement Fraction During Simulation',y=1.01)

    return xfit, np.poly1d(fit.squeeze())

def make_confinement_vs_time_figure(fraction_cyl, fraction_rect, dt=0.01,horizontal=False,t_s=24366,**kwargs):


    norbits = np.linspace(0,fraction_cyl.size*dt,num=fraction_cyl.size)


    if horizontal:
        fig = plt.figure(figsize=(8,3))
        fig.subplots_adjust(left=0.1,bottom=0.19,right=0.98,top=0.91,wspace=0.42)
        ax = fig.add_subplot(121)
        ax.text(-2500,107,"(a)")
    else:
        fig = plt.figure(figsize=(3.37,5))
        fig.subplots_adjust(left=0.25,bottom=0.11,right=0.92,top=0.94,hspace=0.65)
        ax = fig.add_subplot(211)

    ax.plot(norbits,fraction_cyl*100,label='Cylindrical grid',**kwargs)
    ax.plot(norbits,fraction_rect*100,label='Rectangular grid',color=(0.5,0.5,0.5),alpha=0.5)
    ax.set_xlim(0,norbits.max())
    ax.set_ylim(0,100)
    ax.plot([t_s/100,t_s/100],ax.get_ylim(),'k:')
    yfmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(yfmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xticks([0,2500,5000,7500,10000])
    ax.set_xlabel('Number of Fiducial Orbits')
    ax.set_ylabel('Percentage Confined')
    ax.grid()
    ax.set_title('Particle Confinement Fraction',y=1.01)
    ax.legend(loc="upper right",bbox_to_anchor=(1,1),framealpha=1)

    #plot semilogx loss fraction
    if horizontal:
        ax = fig.add_subplot(122)
        ax.text(0.08,107,"(b)")
    else:
        ax = fig.add_subplot(212)

    ax.semilogx(norbits,(1-fraction_cyl)*100,**kwargs)
    ax.semilogx(norbits,(1-fraction_rect)*100,color=(0.5,0.5,0.5),alpha=0.5)
    ax.set_xlim(1,norbits.max())
    ax.set_ylim(0,100)
    ax.plot([t_s/100,t_s/100],ax.get_ylim(),'k:')
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.yaxis.set_minor_formatter(yticks)
    ax.set_xticks([1,10,100,1000,10000])
    ax.set_xlabel('Number of Fiducial Orbits')
    ax.set_ylabel('Percentage Lost')
    ax.grid()
    ax.set_title('Particle Loss Fraction',y=1.01)

    #calculate fit to points after 50% of particles lost
    loss_percent = (1-fraction_cyl)*100
    #start_index = np.where(fraction<0.5)[0][0]
    start_index = t_s
    xfit = np.log10(norbits[start_index:])
    yfit = loss_percent[start_index:]
    fit = np.polyfit(xfit,yfit,1)

    #plot fit
    ax.plot(norbits,fit[0]*np.log10(norbits) + fit[1],'--',color=(1.0,0.6,0),label='Logarithmic fit')
    #ax.text(1.4,80,"Slope$= %2.0f$%% per decade"%fit[0],fontsize=8)
    ax.legend(loc="upper left",bbox_to_anchor=(0,1),framealpha=1)

    return xfit,yfit


def exclude_first_half_lost(max_iterations, total_iter = 1e6, min_iter = 104376): #min_iter for rectangular grid = 36752
    """
    For separating merged summary data into confined and escaped
    while excluding particles that exit during the first bit of the evolution
    when half of the particles escape.

    Returns indices corresponding to only particles that stay confined beyond the first ~370 orbits, which corresponds to the loss of the first half of
    the initial set of particles.

    Returns two subsets of indices based on MAX_ITERATIONS:

    confined, escaped = exclude_first_half_lost(max_iterations)

    The confined indices are only the particles that remain confined for the entire run.  The escaped indices are the particles that escape after the first half of the particles have been lost and before the simulation ends.

    Must use max_iterations derived from a compilation of subruns:
        file_list = ['particle_summary_trajectory_data_6524539_escaped.npz','particle_summary_trajectory_data_7294312_escaped.npz','particle_summary_trajectory_data_10186475_escaped.npz','particle_summary_trajectory_data_10186475_confined.npz']

        r0,rf,v0,B0,max_iterations,is_confined = ostats.collect_subrun_stats(file_list,[0,3e5,6e5,6e5])
    """

    #return arrays of boolean conditions so that logical operations can be performed
    conf_inds = max_iterations == total_iter - 1
    early_inds = max_iterations <= min_iter
    escape_inds = np.logical_and(max_iterations > min_iter, max_iterations < total_iter - 1)

    return conf_inds, escape_inds, early_inds

def exclude_early_inds(max_iterations, total_iter = 1e6, exclude_before = 24367): #min_iter for rectangular grid = 36752
    """
    MODIFIED COPY OF exclude_first_half_lost WITH BETTER NOTATION AND UPDATED DEFAULTS

    For separating merged summary data into confined and escaped
    while excluding particles that exit during the first bit of the evolution
    when half of the particles escape.

    Returns indices corresponding to only particles that stay confined beyond the first MIN_ITER orbits.

    If used for cyl270 data (12/2024), min_iter default of 15600 corresponds to
    point when avg energy crosses initial average energy line in U0 vs confinement plot.

    If used for cyl270 data (12/2024), min_iter default of 24367 corresponds to
    point when fraction confined drops by 1/e.

    Returns two subsets of indices based on MAX_ITERATIONS:

    confined, escaped = exclude_first_half_lost(max_iterations)

    The confined indices are only the particles that remain confined for the entire run.  The escaped indices are the particles that escape after the first half of the particles have been lost and before the simulation ends.

    Must use max_iterations derived from a compilation of subruns:
        file_list = ['particle_summary_trajectory_data_6524539_escaped.npz','particle_summary_trajectory_data_7294312_escaped.npz','particle_summary_trajectory_data_10186475_escaped.npz','particle_summary_trajectory_data_10186475_confined.npz']

        r0,rf,v0,B0,max_iterations,is_confined = ostats.collect_subrun_stats(file_list,[0,3e5,6e5,6e5])
    """

    #return arrays of boolean conditions so that logical operations can be performed
    conf_inds = max_iterations == total_iter - 1
    early_inds = max_iterations <= exclude_before
    escape_inds = np.logical_and(max_iterations > exclude_before, max_iterations < total_iter - 1)

    return conf_inds, escape_inds, early_inds


def prep_for_rect308_histograms():
    """
    Get variables to make histograms from rect308 dataset

    ri, vi, Bi, max_iterations, conf_inds, esc_inds, early_inds = ostats.prep_for_rect308_histograms()
    """
    file_list = [os.path.join(home,data_loc,'particle_summary_trajectory_data_6524539_escaped.npz'),
        os.path.join(home,data_loc,'particle_summary_merge_6524539_7294312_escaped.npz'),
        os.path.join(home,data_loc,'particle_summary_merge_6524539_7294312_10186475_escaped.npz')]
    ri_esc, rf_esc, vi_esc, Bi_esc, max_iterations_esc, not_confined = collect_subrun_stats(file_list,[0,0,0],legacy=True)

    conf_file = os.path.join(home, data_loc, "particle_summary_merge_6524539_7294312_10186475_confined.npz")
    ri_conf, rf_conf, vi_conf, Bi_conf, max_iterations_conf, is_confined = read_summary_data(conf_file,legacy=True)

    ri = np.concatenate([ri_esc,ri_conf])
    rf = None
    vi = np.concatenate([vi_esc,vi_conf])
    Bi = np.concatenate([Bi_esc,Bi_conf])
    max_iterations = np.concatenate([max_iterations_esc,max_iterations_conf])
    is_confined = np.concatenate([not_confined,is_confined])

    conf_inds, escape_inds, early_inds = exclude_early_inds(max_iterations, total_iter = 999998, exclude_before = 36752)
    return ri, vi, Bi, max_iterations, conf_inds, escape_inds, early_inds


def prep_for_cyl270_histograms():
    """
    Get variables to make histograms from cyl270 dataset

    ri, vi, Bi, max_iterations, conf_inds, esc_inds, early_inds = ostats.prep_for_cyl270_histograms()
    """
    conf_file = os.path.join(home, data_loc, "20241218","merged","particle_summary_20241218_merged270pos_confined.npz")
    esc_file = os.path.join(home, data_loc,"20241218","merged","particle_summary_20241218_merged270pos_escaped.npz")
    ri, rf, vi, Bi, max_iterations, is_confined = combine_esc_conf_summaries(esc_file,conf_file,legacy=False)
    conf_inds, escape_inds, early_inds = exclude_early_inds(max_iterations, total_iter = 1e6, exclude_before = 24367)
    return ri, vi, Bi, max_iterations, conf_inds, escape_inds, early_inds


def plot_rL0_histograms(v0, B0, conf_inds, esc_inds, early_inds, ntot=27000, num_bins=700, semilogy=False,title=True,ax=None,legend=True,return_data_only=False, norm_per_bin=False,return_data=False,stacked_bar=False,plot_total=True):
    #make_rL0_histograms(v0, B0, conf_inds, esc_inds, early_inds, ntot=30800, num_bins=700, semilogy=False)

    if ax is None:
        ax = plt.gcf().add_subplot(111)

    rL0_early = tft.get_rL(v0[early_inds],B0[early_inds])
    rL0_esc = tft.get_rL(v0[esc_inds],B0[esc_inds])
    rL0_conf = tft.get_rL(v0[conf_inds],B0[conf_inds])

    num_early,bins_early = np.histogram(rL0_early,bins=num_bins)
    num_esc,bins_esc = np.histogram(rL0_esc,bins=bins_early)
    num_conf,bins_conf = np.histogram(rL0_conf,bins=bins_early)

    if return_data_only:
        return bins_early, num_early, num_esc, num_conf


    bar_width = (bins_conf[-1] - bins_conf[0]) / num_bins

    if semilogy:
        stacking_order = [1, 2, 3]
    else:
        stacking_order = [3, 2, 1]

    if norm_per_bin:
        total_at_each_energy = num_early+num_esc+num_conf
        ntot=total_at_each_energy

    if stacked_bar:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Escaped Early', width=bar_width, log=semilogy, )
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Escaped Later', width=bar_width, log=semilogy, bottom=100*num_early/ntot)
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, bottom=(100*num_esc/ntot + 100*num_early/ntot))
    else:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Escaped Early', width=bar_width, log=semilogy, zorder=stacking_order[0])
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Escaped Later', width=bar_width, log=semilogy, zorder=stacking_order[1])
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, zorder=stacking_order[2])


    if plot_total:
        total_at_each_mu = num_early+num_esc+num_conf
        ax.plot(bins_early[:-1],100*total_at_each_mu/27000,'k--',label='Total')

    ax.set_xlim(0,20)
    ax.set_ylim(1e-4,2.6)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xlabel(r'Initial Gyroradius (in units of $\rho_0$)')
    ax.set_ylabel('Fraction of Particles')
    if title:
        ax.set_title('Distribution of Initial Gyroradius for Escaped and Confined Particles',y=1.03)
    if legend:
        ax.legend(bbox_to_anchor=(0.92,0.9),fontsize=8,framealpha=1)
    ax.grid()

    return bins_early, 100*num_early/ntot, 100*num_esc/ntot, 100*num_conf/ntot

def make_rL0_histograms_figure(v0,B0,conf_inds,esc_inds,early_inds,horizontal=True,ax=None,legend=True,title=True):
    """
    Hard coded to make two-column or preprint (horizontal) version of figure
    """

    if horizontal:
        fig = plt.figure(figsize=(8,3))
        fig.subplots_adjust(left=0.11,bottom=0.19,right=0.98,top=0.9,wspace=0.42)
        #fig.suptitle('Initial Gyroradius Distributions',y=0.99)
        ax = fig.add_subplot(121)
        fig.text(0.01,0.95,"(a)")
        data = plot_rL0_histograms(v0,B0,conf_inds,esc_inds,early_inds,semilogy=False,ax=ax,title=False,stacked_bar=True)
        ax.set_ylim(0,5)
        ax.legend(bbox_to_anchor=(1,1),loc="upper right",fontsize=8,framealpha=1)
        ax.xaxis.set_tick_params(pad=2)
    else:
        fig = plt.figure(figsize=(3.37,5))
        fig.subplots_adjust(left=0.25,bottom=0.11,right=0.92,top=0.94,hspace=0.15)
        ax = fig.add_subplot(211)
        ax.text(-3.5,1.6,"(a)")
        data = plot_rL0_histograms(v0,B0,conf_inds,esc_inds,early_inds,semilogy=False,ax=ax,)
        ax.set_xlabel(None)
        ax.xaxis.set_tick_params(labelcolor='none',pad=1)

    #plot semilogx loss fraction
    if horizontal:
        ax = fig.add_subplot(122)
        fig.text(0.51,0.95,"(b)")
    else:
        ax = fig.add_subplot(212)
        ax.text(-3.5,1.6,"(b)")

    plot_rL0_histograms(v0,B0,conf_inds,esc_inds,early_inds,ax=ax,legend=False,title=False,semilogy=False,plot_total=False,norm_per_bin=True)
    ax.set_ylim(0,100)
    ax.legend(bbox_to_anchor=(0.0,1),loc="upper left",fontsize=8,framealpha=1)
    ax.xaxis.set_tick_params(pad=2)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)



def make_all_rL_histograms(data_dir, smooth=None, file_suffix='escaped.h5', save_file_prefix='rL_counts_',
    nskip=10, total_iter = 1000000, num_bins=100, dt=0.01, exclude_before=24366):
    """
    Gather stats for gyroradius throughout trajectory rather than just initial
    gyroradius.

    EXCLUDE_BEFORE gives index to separate particles that escape early from those that escape later.
    """

    #make a new data file to save 'collision' data
    savefile_name = save_file_prefix+datetime.now().strftime('%Y-%m-%d_hr%Hm%M.npz')

    file_list = []

    for filename in os.listdir(data_dir):
        if filename.endswith(file_suffix):
            file_list.append(filename)
    nfiles = len(file_list)
    start_time = time.time()


    bin_edges = np.linspace(0,rLmax,num_bins+1,)
    num_particles = 0

    for filecount, filename in enumerate(file_list):
        file_time = time.time()
        with h5py.File(os.path.join(data_dir,filename),'r') as file_obj:
            print('Reading data file {} of {}: '.format(filecount+1,nfiles) + filename)
            fkeys = list(file_obj.keys())
            for vindex in fkeys:
                #get trajectory and iterations
                #r = file_obj[vindex]['r'][()]
                v = file_obj[vindex]['v'][()]
                B = file_obj[vindex]['B'][()]
                iter = file_obj[vindex]['iter'][()]

                #calculate gyroradius and min gradient scale length
                rL = tft.get_rL(v,B)

                if smooth is not None:
                    rL = mytools.smooth(rL,window_len=smooth)
                else:
                    smooth = 0

                tempiters_early, bin_edges = np.histogram(rL_early[rL_early>0], bins=bin_edges,)

                if num_particles == 0:
                    num_orbits = tempiters*dt
                else:
                    num_orbits += tempiters*dt

                num_particles += 1

        #save dataset as is and overwrite with more complete data next iteration of loop (next data file)
        np.savez(savefile_name, num_orbits=num_orbits, bin_edges=bin_edges, total_iter=total_iter, rLmax=rLmax, data_dir=data_dir, smooth=smooth, file_suffix=file_suffix, num_particles=num_particles, exclude_first_half_lost=exclude_first_half_lost)
        print("File read time: {}\n".format(time.time()-file_time))

    print("Total run time: {}\n".format(time.time()-start_time))
    return num_orbits, bin_edges, total_iter, num_particles



def plot_U0_histograms(v0,conf_inds, esc_inds, early_inds, ntot=27000,num_bins=250,semilogy=False, t_s=24367, ax=None, legend=True, title=True, norm_per_bin=False,return_data=False,stacked_bar=False,plot_total=True):
    #plot_U0_histograms(v0,conf_inds, esc_inds, early_inds, ntot=30800,num_bins=150,semilogy=False) #rectangular

    if ax is None:
        ax = plt.gcf().add_subplot(111)

    U0_early = (0.5*v0[early_inds]**2).sum(axis=-1)
    U0_conf = (0.5*v0[conf_inds]**2).sum(axis=-1)
    U0_esc = (0.5*v0[esc_inds]**2).sum(axis=-1)

    num_early,bins_early = np.histogram(U0_early,bins=num_bins)
    num_esc,bins_esc = np.histogram(U0_esc,bins=bins_early)
    num_conf,bins_conf = np.histogram(U0_conf,bins=bins_early)

    bar_width = (bins_conf[-1] - bins_conf[0]) / num_bins

    if semilogy:
        stacking_order = [1, 2, 3]
    else:
        stacking_order = [3, 2, 1]

    if norm_per_bin:
        total_at_each_energy = num_early+num_esc+num_conf
        ntot=total_at_each_energy

    if stacked_bar:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Escaped Early', width=bar_width, log=semilogy, )
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Escaped Later', width=bar_width, log=semilogy, bottom=100*num_early/ntot)
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, bottom=(100*num_esc/ntot + 100*num_early/ntot))
    else:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Escaped Early', width=bar_width, log=semilogy, zorder=stacking_order[0])
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Escaped Later', width=bar_width, log=semilogy, zorder=stacking_order[1])
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, zorder=stacking_order[2])


    if plot_total:
        total_at_each_mu = num_early+num_esc+num_conf
        ax.plot(bins_early[:-1],100*total_at_each_mu/27000,'k--',label='Total')


    #ax.set_ylim(0,1.6)
    ax.set_xlim(0,5)
    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xlabel('Particle Energy ($kT$)')
    ax.set_ylabel('Fraction of Particles')

    if title:
        ax.set_title('Particle Energy Distributions',y=1.01)

    if legend:
        ax.legend(bbox_to_anchor=(1,1),loc='upper right',fontsize=8,framealpha=1)
    ax.grid()
    #ax.plot(norbits,np.ones(norbits.shape)*33.33,'k--')
    if return_data:
        return bins_early, num_early, num_esc, num_conf


def make_U0_histograms_figure(v0,conf_inds,esc_inds,early_inds,horizontal=True):
    """
    Hard coded to make two-column or preprint (horizontal) version of figure
    """

    if horizontal:
        fig = plt.figure(figsize=(8,3))
        fig.subplots_adjust(left=0.11,bottom=0.19,right=0.99,top=0.9,wspace=0.42)
        #fig.suptitle('Particle Energy Distributions',y=0.99)
        ax = fig.add_subplot(121)
        fig.text(0.01,0.95,"(a)")
        plot_U0_histograms(v0,conf_inds,esc_inds,early_inds,semilogy=False,ax=ax,title=False,stacked_bar=True,plot_total=True)
        ax.set_ylim(0,3.5)
    else:
        fig = plt.figure(figsize=(3.37,5))
        fig.subplots_adjust(left=0.25,bottom=0.11,right=0.92,top=0.94,hspace=0.4)
        ax = fig.add_subplot(211)
        ax.text(-1.5,1.4,"(a)")
        plot_U0_histograms(v0,conf_inds,esc_inds,early_inds,semilogy=False,ax=ax,title=False,stacked_bar=True,plot_total=True)

    #plot semilogx loss fraction
    if horizontal:
        ax = fig.add_subplot(122)
    else:
        ax = fig.add_subplot(212)

    fig.text(0.51,0.95,"(b)")
    plot_U0_histograms(v0,conf_inds,esc_inds,early_inds,ax=ax,legend=False,title=False,norm_per_bin=True,stacked_bar=False,plot_total=False)
    ax.legend(bbox_to_anchor=(0.05,1),loc='upper left',fontsize=8,framealpha=1)
    ax.set_ylabel("Fraction of Particles with Each $U_0$")
    ax.set_ylim(0,70)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)

def plot_mu0_histograms(v0, B0, conf_inds, esc_inds, early_inds, ntot=27000, num_bins=100, xlim=[0,20], ylim=[0,10], clip=True, semilogy=False,ax=None,legend=True,title=True,norm_per_bin=False,return_data=False,stacked_bar=False,plot_total=True):
    #plot_mu0_histograms(v0, B0, conf_inds, esc_inds, early_inds, ntot=30800, num_bins=300, semilogy=False)
    #log version: plot_mu0_histograms(vi,Bi,conf_inds,esc_inds,early_inds,xlim=[0,50],ylim=[0.01,15],semilogy=True,num_bins=100)

    if ax is None:
        fig = plt.figure(figsize=(3.37,2.5))
        fig.subplots_adjust(left=0.23,bottom=0.22,right=0.95,top=0.88,)
        ax = fig.add_subplot(111)

    # since variables are a sequence of initial conditions for all particles,
    # get___along_traj returns the variable for each set of initial conditions
    mu0_early = tft.get_mu_along_traj(v0[early_inds],B0[early_inds])
    if clip:
        print("Actual max $\mu$ is {}".format(mu0_early.max()))
        print("Clipping to {}".format(xlim))
        mu0_early = np.clip(mu0_early,xlim[0],xlim[1]+1)

    mu0_esc = tft.get_mu_along_traj(v0[esc_inds],B0[esc_inds])
    mu0_conf = tft.get_mu_along_traj(v0[conf_inds],B0[conf_inds])

    num_early,bins_early = np.histogram(mu0_early,bins=num_bins,range=xlim)
    num_esc,bins_esc = np.histogram(mu0_esc,bins=bins_early,range=xlim)
    num_conf,bins_conf = np.histogram(mu0_conf,bins=bins_early,range=xlim)


    #max_mu = np.max([bins[-1],bins_esc[-1]]))
    bar_width = (bins_conf[-1] - bins_conf[0]) / num_bins

    #plot as percent, so calculate fraction of total in each bin * 100

    if semilogy:
        stacking_order = [1, 2, 3]
    else:
        stacking_order = [3, 2, 1]

    if norm_per_bin:
        total_at_each_mu = num_early+num_esc+num_conf
        ntot=total_at_each_mu

    if stacked_bar:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Escaped Early', width=bar_width, log=semilogy, )
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Escaped Later', width=bar_width, log=semilogy, bottom=100*num_early/ntot)
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, bottom=(100*num_esc/ntot + 100*num_early/ntot))
    else:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Escaped Early', width=bar_width, log=semilogy, zorder=stacking_order[0])
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Escaped Later', width=bar_width, log=semilogy, zorder=stacking_order[1])
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, zorder=stacking_order[2])


    if plot_total:
        total_at_each_mu = num_early+num_esc+num_conf
        ax.plot(bins_early[:-1],100*total_at_each_mu/27000,'k--',label='Total')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xlabel('Initial Magnetic Moment ($kT/B_0$)')
    ax.set_ylabel('Fraction of Particles')
    if title:
        ax.set_title('Distribution of Initial $\mu$',y=1.03)

    if legend:
        ax.legend(bbox_to_anchor=(1,1),loc='upper right',fontsize=8)
    ax.grid()
    #ax.plot(norbits,np.ones(norbits.shape)*33.33,'k--')
    if return_data:
        return bins_early, num_early, num_esc, num_conf


def make_mu0_histograms_figure(v0,B0,conf_inds,esc_inds,early_inds,horizontal=True,ax=None,legend=True,title=True,):
    """
    Hard coded to make two-column or preprint (horizontal) version of figure
    """

    if horizontal:
        fig = plt.figure(figsize=(8,3))
        fig.subplots_adjust(left=0.11,bottom=0.19,right=0.98,top=0.9,wspace=0.42)
        #fig.suptitle('Magnetic Moment Distributions',y=0.99)
        ax = fig.add_subplot(121)
        ax.text(-6,107,"(a)")
        plot_mu0_histograms(v0,B0,conf_inds,esc_inds,early_inds,semilogy=False,ylim=[0,100],ax=ax,title=False,norm_per_bin=True,stacked_bar=True,plot_total=False)
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        ax.xaxis.set_tick_params(pad=2)
        ax.set_ylim(0,100)
        ax.set_ylabel("Fraction of Particles with Each $\mu$")
        ax.legend(bbox_to_anchor=(1,0),loc='lower right',fontsize=8)
    else:
        fig = plt.figure(figsize=(3.37,5))
        fig.subplots_adjust(left=0.25,bottom=0.11,right=0.92,top=0.94,hspace=0.15)
        ax = fig.add_subplot(211)
        ax.text(-3.5,3.05,"(a)")
#        plot_mu0_histograms(v0,B0,conf_inds,esc_inds,early_inds,semilogy=False,ylim=[0,100],ax=ax,norm_per_bin=True)
        ax.set_xlabel(None)
        ax.xaxis.set_tick_params(labelcolor='none',pad=1)

    #plot semilogx loss fraction
    if horizontal:
        ax = fig.add_subplot(122)
    else:
        ax = fig.add_subplot(212)

    if horizontal:
        ax.text(-6,16,"(b)")
    else:
        ax.text(-3.5,3,"(b)")
    plot_mu0_histograms(v0,B0,conf_inds,esc_inds,early_inds,semilogy=True,ylim=[1e-2,10],ax=ax,legend=True,title=False)
    fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.xaxis.set_tick_params(pad=2)


def plot_pitch_histograms(v0, B0, conf_inds, esc_inds, early_inds, ntot=27000, num_bins=100, xlim=[0,np.pi], ylim=[0,1], semilogy=False,ax=None,legend=True,title=False,return_data=False,norm_per_bin=False,plot_total=False,stacked_bar=True):
    #plot_mu0_histograms(v0, B0, conf_inds, esc_inds, early_inds, ntot=30800, num_bins=300, semilogy=False)
    #log version: plot_mu0_histograms(vi,Bi,conf_inds,esc_inds,early_inds,xlim=[0,50],ylim=[0.01,15],semilogy=True,num_bins=100)

    if ax is None:
        fig = plt.figure(figsize=(3.37,2.5))
        fig.subplots_adjust(left=0.2,bottom=0.18,right=0.98,top=0.97,)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(pad=2, labelsize=8)
        ax.yaxis.set_tick_params(pad=2, labelsize=8)

    # since variables are a sequence of initial conditions for all particles,
    # get___along_traj returns the variable for each set of initial conditions
    theta0_early = get_pitch_angle(v0[early_inds],B0[early_inds])
    theta0_esc = get_pitch_angle(v0[esc_inds],B0[esc_inds])
    theta0_conf = get_pitch_angle(v0[conf_inds],B0[conf_inds])

    num_early,bins_early = np.histogram(theta0_early,bins=num_bins,range=xlim)
    num_esc,bins_esc = np.histogram(theta0_esc,bins=bins_early,range=xlim)
    num_conf,bins_conf = np.histogram(theta0_conf,bins=bins_early,range=xlim)


    #max_theta = np.max([bins[-1],bins_esc[-1]]))
    bar_width = (bins_conf[-1] - bins_conf[0]) / num_bins

    #plot as percent, so calculate fraction of total in each bin * 100

    if semilogy:
        stacking_order = [1, 2, 3]
    else:
        stacking_order = [3, 2, 1]

    if norm_per_bin:
        total_at_each_angle = num_early+num_esc+num_conf
        ntot=total_at_each_angle

    if stacked_bar:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Esc. Early', width=bar_width, log=semilogy, zorder=stacking_order[0])
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Esc. Later', width=bar_width, log=semilogy, zorder=stacking_order[1],bottom=100*num_early/ntot)
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, zorder=stacking_order[2],bottom=(100*num_esc/ntot + 100*num_early/ntot))
    else:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Esc. Early', width=bar_width, log=semilogy, zorder=stacking_order[0])
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Esc. Later', width=bar_width, log=semilogy, zorder=stacking_order[1],)
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, zorder=stacking_order[2],)


    if plot_total:
        total_at_each_angle = num_early+num_esc+num_conf
        ax.plot(bins_early[:-1],100*total_at_each_angle/27000,'k--',label='Total')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xlabel(r'Initial Pitch Angle $\theta_0$ (rad)')
    ax.set_ylabel('Fraction of Particles')
    if title:
        ax.set_title('Distribution of Initial Pitch Angle',y=1.03)

    if legend:
        ax.legend(bbox_to_anchor=(-0.01,1.01),loc='upper left',fontsize=8,framealpha=1)
    ax.grid()
    #ax.plot(norbits,np.ones(norbits.shape)*33.33,'k--')
    if return_data:
        return bins_early, num_early, num_esc, num_conf


def make_pitch_histograms_figure(v0,B0,conf_inds,esc_inds,early_inds):
    """
    Hard coded to make two-column or preprint (horizontal) version of figure
    """
    fig = plt.figure(figsize=(8,3))
    fig.subplots_adjust(left=0.11,bottom=0.19,right=0.98,top=0.89,wspace=0.42)
    #fig.suptitle('Magnetic Moment Distributions',y=0.99)
    ax = fig.add_subplot(121)
    plot_pitch_histograms(v0,B0,conf_inds,esc_inds,early_inds,semilogy=False,ylim=[0,2.5],ax=ax,legend=True,plot_total=True,stacked_bar=True)
    ax.xaxis.set_tick_params(pad=2)
    ax.text(-0.85,2.73,"(a)")
    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)

    ax = fig.add_subplot(122)
    plot_pitch_histograms(v0,B0,conf_inds,esc_inds,early_inds,semilogy=False,ylim=[0,100],ax=ax,title=False,norm_per_bin=True,legend=False,stacked_bar=False)
    ax.xaxis.set_tick_params(pad=2)
    ax.set_ylim(0,100)
    ax.text(-1,109,"(b)")
    ax.set_ylabel(r"Fraction of Particles with Each $\theta_0$")
    ax.legend(bbox_to_anchor=(0.295,1),loc='upper left',fontsize=8,framealpha=1)
    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)


def get_avgU0_vs_conf_time(v0,max_iterations,conf_inds,kernel_size=100):
    """
    Calculate moving average of particle energy vs confinement time
    using Gaussian kernel with sigma = kernel_size
    """

    all_esc_inds = np.logical_not(conf_inds)
    U0 = (0.5*v0[all_esc_inds]**2).sum(axis=-1)
    conf_iter = max_iterations[all_esc_inds]
    sort_inds = np.argsort(max_iterations[all_esc_inds])

    U0 = U0[sort_inds]
    conf_iter = conf_iter[sort_inds]

    return conf_iter, mytools.gsmooth(U0,kernel_size)


def plot_avgU0_vs_conf_time(v0, max_iterations, conf_inds, kernel_size=100,
        dt=0.01,semilogx=False, U0_conf=1.130, U0_initial=1.484, exclude_orbits=243.67):

    #rectangular grid 308 positions
    #plot_avgU0_vs_conf_time(v0, max_iterations, conf_inds, kernel_size=100, dt=0.01,semilogx=True, U0_conf=1.066, half_gone_orbits=367.52):

    conf_iter, smU0_avg = get_avgU0_vs_conf_time(v0, max_iterations, conf_inds, kernel_size=kernel_size)
    conf_iter, U0_avg = get_avgU0_vs_conf_time(v0, max_iterations, conf_inds, kernel_size=1)

    fig = plt.figure(figsize=(5,3))
    fig.subplots_adjust(left=0.1,bottom=0.15,right=0.95,top=0.89)
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=8)
    if semilogx:
        ax.semilogx(conf_iter*dt,U0_avg,label='Escaping particles',color='b',alpha=0.3)
        ax.semilogx(conf_iter*dt,smU0_avg,label='Running average',color='b',zorder=5)
    else:
        ax.plot(conf_iter*dt,U0_avg,label='Escaping particles',color='b',alpha=0.3)
        ax.plot(conf_iter*dt,smU0_avg,label='Running average',color='b',zorder=5)
    ax.set_ylim(0,7)
    ax.set_xlim(-50,1e4)
    ax.set_ylabel(r'$U_i$ ($kT$)',fontsize=9)
    ax.set_xlabel('Confinement Time (fiducial orbits)',fontsize=9)
    ax.set_title('Average Energy of Escaping Particles',y=1.01,fontsize=10)
    ax.plot(ax.get_xlim(),[U0_initial,U0_initial],'k-.',label='Initial average')
    ax.plot(ax.get_xlim(),[U0_conf,U0_conf],'k--',label='Confined average')
    ax.plot([exclude_orbits,exclude_orbits],ax.get_ylim(),color='k',linestyle=':')#label='$t_{s}$'
    ax.legend(bbox_to_anchor=(0.7,1),loc='upper right',fontsize=8)

    #ax.grid()
    #ax.plot(norbits,np.ones(norbits.shape)*33.33,'k


def get_avg_rL0_vs_conf_time(v0,B0,max_iterations,conf_inds,kernel_size=100):
    """
    Calculate moving average of particle energy vs confinement time
    using Gaussian kernel with sigma = kernel_size
    """

    all_esc_inds = np.logical_not(conf_inds)
    rL0 = tft.get_rL(v0[all_esc_inds],B0[all_esc_inds])
    conf_iter = max_iterations[all_esc_inds]
    sort_inds = np.argsort(max_iterations[all_esc_inds])

    rL0 = rL0[sort_inds]
    conf_iter = conf_iter[sort_inds]

    return conf_iter, mytools.gsmooth(rL0,kernel_size)

def plot_avg_rL0_vs_conf_time(conf_iter, rL0_avg, dt=0.01,semilogx=False, rL0_conf=2.27, half_gone_orbits=243.67):

    ax = plt.gcf().add_subplot(111)
    if semilogx:
        ax.semilogx(conf_iter*dt,rL0_avg,label='Escaped')
    else:
        ax.plot(conf_iter*dt,rL0_avg,label='Escaped')
    ax.set_ylim(0,20)
    ax.set_xlim(-100,1e4)
    ax.set_ylabel('Average Gyroradius (in units of $m_p v_{th}/qB_0$)')
    ax.set_xlabel('Confinement Time (in units of fiducial orbits)')
    ax.set_title('Average Gyroradius of Escaping Particles vs Confinement Time',y=1.01)
    ax.plot(ax.get_xlim(),[rL0_conf,rL0_conf],'k--',label='Confined')
    ax.plot([half_gone_orbits,half_gone_orbits],ax.get_ylim(),'k:',)
    ax.legend(bbox_to_anchor=(0.875,0.93))
    #ax.grid()
    #ax.plot(norbits,np.ones(norbits.shape)*33.33,'k


def get_avg_mu0_vs_conf_time(v0,B0,max_iterations,conf_inds,kernel_size=100):
    """
    Calculate moving average of particle energy vs confinement time
    using Gaussian kernel with sigma = kernel_size
    """

    all_esc_inds = np.logical_not(conf_inds)
    mu0 = tft.get_mu_along_traj(v0[all_esc_inds],B0[all_esc_inds])
    conf_iter = max_iterations[all_esc_inds]
    sort_inds = np.argsort(max_iterations[all_esc_inds])

    mu0 = mu0[sort_inds]
    conf_iter = conf_iter[sort_inds]

    return conf_iter, mytools.gsmooth(mu0,kernel_size)

def plot_avg_mu0_vs_conf_time(conf_iter, mu0_avg, dt=0.01,semilogx=True, mu0_conf=2.135, half_gone_orbits=367.52):

    ax = plt.gcf().add_subplot(111)
    if semilogx:
        ax.semilogx(conf_iter*dt,mu0_avg,label='Escaped')
    else:
        ax.plot(conf_iter*dt,mu0_avg,label='Escaped')
    ax.set_ylim(0,50)
    ax.set_xlim(-100,1e4)
    ax.set_ylabel('Average Magnetic Moment (in units of $m_p v_{th}^2/B_0$)')
    ax.set_xlabel('Confinement Time (in units of fiducial orbits)')
    ax.set_title('Average Magnetic Moment of Escaping Particles vs Confinement Time',y=1.01)
    ax.plot(ax.get_xlim(),[mu0_conf,mu0_conf],'k--',label='Confined')
    ax.plot([half_gone_orbits,half_gone_orbits],ax.get_ylim(),'k:',)
    ax.legend(bbox_to_anchor=(0.875,0.93))
    #ax.grid()
    #ax.plot(norbits,np.ones(norbits.shape)*33.33,'k


def plot_mu_vs_phase_space_traj10000000a(r=None,v=None, B=None):
    if (r is None) or (v is None) or (B is None):
        dnafile = "../traj10000000a_dt0p01_[-55,_55,_425]_[0.1,_-0.8,_1.0].h5"
        r,v,B,iterations,is_confined = read_single_h5_trajectory(dnafile,)
    nmin=0
    nmax=55000
    nskip=2
    mu = tft.get_mu_along_traj(v[nmin:nmax:nskip],B[nmin:nmax:nskip])
    x = r[nmin:nmax:nskip,0]
    vx = v[nmin:nmax:nskip,0]

    fig = plt.figure(figsize=(5,3))
    fig.subplots_adjust(left=0.12,bottom=0.15,right=0.90,top=0.95)
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=8)
    pcollection = ax.scatter(x,vx,c=mu,cmap=plt.cm.viridis)
    ax.set_xlabel(r'$x$ (${\rho_0}$)')
    ax.set_ylabel('$v_x$ ($v_{th}$)')

    divider = make_axes_locatable(ax)
    cbkwargs=dict(position="right",size="5%",pad=0.1)
    cax = divider.append_axes(**cbkwargs)
    cb = plt.colorbar(pcollection,cax=cax,cmap=plt.cm.viridis,)
    cb.set_label("Magnetic moment ($m_p v_{th}^2/B_0$)")

def plot_mu_vs_time_traj10000000a(r=None,v=None, B=None, zlim=[340,420],return_data=False):
    if (r is None) or (v is None) or (B is None):
        dnafile = "../traj10000000a_dt0p01_[-55,_55,_425]_[0.1,_-0.8,_1.0].h5"
        r,v,B,iterations,is_confined = read_single_h5_trajectory(dnafile,)
    #nmin=0
    #nmax=55000
    nmin=0
    nmax=105000
    nskip=2
    mu = tft.get_mu_along_traj(v[nmin:nmax:nskip],B[nmin:nmax:nskip])
    t = 0.01*np.arange(nmin,nmax,nskip)

    fig = plt.figure(figsize=(5,3))
    fig.subplots_adjust(left=0.12,bottom=0.15,right=0.90,top=0.95)
    ax = fig.add_subplot(111)#(211)  #doing subplots here din't work because scaling of 3D plot couldn't fill space effectively
    ax.tick_params(labelsize=8)
    pcollection = ax.scatter(t,mu,c=t,cmap=plt.cm.viridis,marker=',',s=0.2,label='Instantaneous')
    avg_mu = tft.gaussian_filter1d(mu,sigma=11*100/nskip)
    ax.plot(t,avg_mu,'k',label='Gyroaverage')
    ax.set_xlabel(r'$t$ (${\tau_0}$)')
    ax.set_ylabel('$\mu$ ($m_p v_{th}^2/B_0$)')
    ax.set_ylim(0,7)
    ax.set_xlim(nmin/100,nmax/100)
    #ax.legend(scatterpoints=3)
    if return_data:
        return t, mu, avg_mu
    plt.savefig("fig16a_mu_vs_t.png",dpi=600,bbox_inches='tight')

    #print(r[nmin:nmax:nskip,2].min(),r[nmin:nmax:nskip,2].max())

    #ax = fig.add_subplot(212, projection='3d')
    fig = plt.figure(figsize=(7,9))
    ax = fig.add_subplot(111, projection='3d')
    tft.generate_cylinder((0, 0, zlim[0]), (0, 0, zlim[1]), 100, ax=ax)
    ax.set_proj_type('persp', focal_length = 0.7)
    tft.set_axes_equal(ax)
    ax.set_xlim(-110, 110)
    ax.set_ylim(-120, 120)
    ax.set_zlim(zlim)
    #ax.set_zlim(300, 500)
    #ax.view_init(225,91,-90)
    ax.view_init(-164,95,-92)
    ax.tick_params(labelsize=24,pad=10)
    tft.time_as_color_3D(r[nmin:nmax:nskip],ax=ax,nskip=5,cmap=plt.cm.viridis,marker=',',s=0.01)
    ax.set_xlabel(r'$x$ ($\rho_0$)',size=28,labelpad=12)
    ax.set_ylabel(r'$y$ ($\rho_0$)',size=28,labelpad=15)
    ax.set_yticks([-100,0,100])
    ax.set_zlabel(r'$z$ ($\rho_0$)',size=28,labelpad=12)
    ax.set_zticks([350,375,400])

    ax.set_position([0.2,0.1,0.8,0.7])
    ax.set_box_aspect((2,2,1), zoom=1.6)
    plt.savefig("fig16b_mu_vs_t.png",dpi=600,)
    #ax.set_ylim(0,7)
    #ax.set_xlim(0,555)

    #divider = make_axes_locatable(ax)
    #cbkwargs=dict(position="right",size="5%",pad=0.1)
    #cax = divider.append_axes(**cbkwargs)
    #cb = plt.colorbar(pcollection,cax=cax,cmap=plt.cm.viridis,)
    #cb.set_label("Magnetic moment ($m_p v_{th}^2/B_0$)")



def save_positions_for_VisIt(rlist,clist=None, savefile='positions_for_VisIt.txt'):
    """
    Applies VisIt scaling (1 x 1 x 10 units) to list of grid positions
    and saves as text file so that points can be read into VisIt for a
    scatter plot.

    Can also pass an optional color array, CLIST with the same length.
    """
    if clist is not None:
        textarr = np.hstack((np.array(rlist)/100., np.array(clist).reshape(len(clist),1)))
    else:
        textarr = np.array(rlist)/100.

    np.savetxt(savefile,textarr,delimiter=',')


def count_escape_before_t50_by_position(r0,max_iter,all_positions,iter_t50=36752,save_text_file=False,
    savefile='position_counts.txt',visit_scaling=True):
    """
    Counts the frequency of each initial position for particles
    that escape before the time at which half the particles have left
    the volume (t50).

    ALL_POSITIONS is an optional list of all possible initial positions.
    If this parameter is provided, any positions not in R0LIST will be added
    along with a 0 in the nlist output.
    """

    escaped_early = max_iter < iter_t50
    r0list = []
    nlist = []
    frequency_list = Counter(tuple(i) for i in r0[escaped_early])

    for key,value in frequency_list.items():
        r0list.append(list(key))
        #radlist.append( sqrt((r0_i[0:2]**2).sum()) )
        nlist.append(value)

    if all_positions is not None:
        #fill in every position, including places where there are zero counts
        for initial_pos in all_positions:

            if list(initial_pos) not in r0list:
                #if this position doesn't show up in the list of positions
                # in the particle summary data, then add it in with a zero
                # frequency of appearance.
                r0list.append(list(initial_pos))
                nlist.append(0)

    if save_text_file:
        if visit_scaling:
            # x = np.array(r0list)[:,0]
            # y = np.array(r0list)[:,0]
            # z = np.array(r0list)[:,0]
            textarr = np.hstack((np.array(r0list)/100.,np.array(nlist).reshape(len(nlist),1)))

        else:
            textarr = np.hstack((np.array(r0list),np.array(nlist).reshape(len(nlist),1)))

        np.savetxt(savefile,textarr,delimiter=',')

    return np.array(r0list), np.array(nlist)



def count_confined_by_position(r0,save_text_file=False,all_positions=None,
    savefile='position_counts.txt',visit_scaling=True):
    """
    Counts the frequency of each initial position for particles
    that are in particle summary data (confined or escaped)

    ALL_POSITIONS is an optional list of all possible initial positions.
    If this parameter is provided, any positions not in R0LIST will be added
    along with a 0 in the nlist output.
    """

    r0list = []
    nlist = []
    frequency_list = Counter(tuple(i) for i in r0)

    for key,value in frequency_list.items():
        r0list.append(list(key))
        #radlist.append( sqrt((r0_i[0:2]**2).sum()) )
        nlist.append(value)

    if all_positions is not None:
        #fill in every position, including places where there are zero counts
        for initial_pos in all_positions:

            if list(initial_pos) not in r0list:
                #if this position doesn't show up in the list of positions
                # in the particle summary data, then add it in with a zero
                # frequency of appearance.
                r0list.append(list(initial_pos))
                nlist.append(0)

    if save_text_file:
        if visit_scaling:
            # x = np.array(r0list)[:,0]
            # y = np.array(r0list)[:,0]
            # z = np.array(r0list)[:,0]
            textarr = np.hstack((np.array(r0list)/100.,np.array(nlist).reshape(len(nlist),1)))

        else:
            textarr = np.hstack((np.array(r0list),np.array(nlist).reshape(len(nlist),1)))

        np.savetxt(savefile,textarr,delimiter=',')

    return np.array(r0list), np.array(nlist)

def generate_visit_percent_confined_by_position(conf_file):
    """
    Builds visit plot data for percentage of particles confined at each position

    conf_file = "merged/particle_summary_20241218_merged270pos_confined.npz"
    """
    xx,yy,zz = pr.set_up_grid_polar(nr=5, ntheta=9, nz=6, rmin=10, rmax=90, zmin=10, zmax=500, show_plot=False)
    position_list = list(zip(xx,yy,zz))

    ri, rf, vi, Bi, max_iterations_cyl270, is_confined = read_summary_data(conf_file)
    r0list, nlist = count_confined_by_position(ri, save_text_file=True, savefile='position_counts_cyl270.txt', all_positions=position_list)
    return r0list, nlist

def get_iter_vs_position(r0, max_iter, allr0, dt=0.01,save_text_file=False,
    savefile='position_iter_counts.txt', visit_scaling=True):

    n_at_pos = []
    pos_avg_orbits = []
    pos_max_orbits = []
    max_orbits = max_iter*dt
    uniq_r0 = np.unique(allr0,axis=0)

    for ii, uniq_pos in enumerate(uniq_r0):
        pos_indices = (r0==uniq_pos).all(axis=1)

        #if all false, put in a zero rather than try to calculate the mean of no elements
        if pos_indices.sum()==0:
            pos_avg_orbits.append(0)
            pos_max_orbits.append(0)
        else:
            pos_avg_orbits.append(max_orbits[pos_indices].mean())
            pos_max_orbits.append(max_orbits[pos_indices].max())

    #print(len(pos_avg_orbits))
    #print(len(uniq_r0))
    #return pos_avg_orbits, pos_avg_orbits, pos_avg_orbits
    pos_avg_orbits = np.array(pos_avg_orbits).reshape(len(uniq_r0),1)
    pos_max_orbits = np.array(pos_max_orbits).reshape(len(uniq_r0),1)

    if save_text_file:
        if visit_scaling:
            # x = np.array(r0list)[:,0]
            # y = np.array(r0list)[:,0]
            # z = np.array(r0list)[:,0]
            textarr = np.hstack([uniq_r0/100.,pos_avg_orbits,pos_max_orbits])

        else:
            textarr = np.hstack([uniq_r0,pos_avg_orbits,pos_max_orbits])

        np.savetxt(savefile,textarr,delimiter=',')

    return uniq_r0, pos_avg_orbits, pos_max_orbits



def generate_visit_avg_iter_by_position(esc_file,conf_file, allr0, exclude_before=15600,
        savefile="position_avg_iter_excl_early.txt", exclude_after=False, **kwargs):

    ri, rf, vi, Bi, max_iterations, is_confined = combine_esc_conf_summaries(esc_file,conf_file,legacy=True)
    conf_inds, escape_inds, early_inds = exclude_early_inds(max_iterations,exclude_before=exclude_before)
    #visit_scaling, divide by 100:

    if exclude_after:
        uniq_r0, pos_avg_orbits, pos_max_orbits = get_iter_vs_position(ri[early_inds],
                                                    max_iterations[early_inds],
                                                    allr0,
                                                    save_text_file=True,
                                                    savefile=savefile)
    else:
        later_inds = np.logical_or(conf_inds, escape_inds)
        uniq_r0, pos_avg_orbits, pos_max_orbits = get_iter_vs_position(ri[later_inds],
                                                    max_iterations[later_inds],
                                                    allr0,
                                                    save_text_file=True,
                                                    savefile=savefile)

    return uniq_r0, pos_avg_orbits, pos_max_orbits



def count_confined_by_radius(r,max_iter,zval=500,iter_snapshot=36752,):
    """
    Counts the confinement fraction at a given number of iterations as a
    function of radius from particle summary data.

    R0 = array of initial positions with shape (N,3) that are included in
    given summary dataset

    max_iter = list of how long each particle was confined

    In [69]: unique(z)
    Out[69]: array([ 10.,  91., 173., 255., 336., 418., 500.])

    """

    #convert to polar coordinates
    # x,y,z = r0.transpose()
    # r = np.sqrt(x[z==zval]**2 + y[z==zval]**2)
    # ziter = max_iter[z==zval]

    uniq_r = np.unique(r)
    avg_iter = []

    for eachr in uniq_r:
        condition = r==eachr
        avg_iter.append(max_iter[condition].mean())

    return uniq_r, np.array(avg_iter)

def get_avg_iter_vs_r_z(r0, max_iter, dt=0.01,):

    x,y,z = r0.transpose()
    r = np.sqrt(x**2 + y**2)

    uniq_r = np.unique(r)
    uniq_z = np.unique(z)

    avg_orbits = np.zeros(shape=(uniq_r.size,uniq_z.size))
    max_orbits = max_iter*dt

    for ii,zval in enumerate(uniq_z):
        uniq_r, avg_orbits[:,ii] = count_confined_by_radius(r[z==zval], max_orbits[z==zval])

        ru, zu = np.meshgrid(uniq_r, uniq_z)

    return ru, zu, avg_orbits.transpose()


def plot_confined_by_radius_z(r0, max_iter, dt=0.01, dim=2, **kwargs):
    x,y,z = r0.transpose()
    r = np.sqrt(x**2 + y**2)
    mycolors = mytools.RB_linear()
    uniq_z = np.unique(z)
    nz = uniq_z.size

    fig = plt.gcf()
    plt.clf()
    if dim == 3:
        ax = fig.add_subplot(111,projection='3d')
    else:
        ax = fig.add_subplot(111)

    max_orbits = max_iter*dt

    for ii,zval in enumerate(uniq_z):
        uniq_r, avg_orbits = count_confined_by_radius(r[z==zval], max_orbits[z==zval])
        if dim == 3:
            ax.plot(uniq_r, zval*np.ones(uniq_z.size), zs=avg_orbits, label=zval, color=mycolors(ii/nz),**kwargs)
            ax.set_xlabel('Initial Radius ($r/\\rho_{th}$)')
            ax.set_zlabel('Average thermal orbits confined')
            ax.set_ylabel('Initial Z position ($z/\\rho_{th}$)')
        else:
            ax.plot(uniq_r, avg_orbits, label=zval, color=mycolors(ii/nz),**kwargs)
            ax.set_xlabel('Initial Radius ($r/\\rho_{th}$)')
            ax.set_ylabel('Average number of orbits confined')

    ax.set_title('Average Confinement Times', y=1.02)
    return ax




def get_gradient_collision_stats(gradLmin, data_dir, file_suffix='escaped.h5', save_file_prefix='collision_fractions_',
                                nskip=100, rL_thresh=5, exclude_before=None, exclude_after=None, total_iter=1000000):
    """
    Recover number of iterations completed before escape
    and the fraction of time during the run that the particle
    was in a region where the minimum (tensor) gradient scale length
    was less than the current gyroradius.
    Also count number of "collisions"--  the number of times during the run that the particle
    crossed into or out of a region where the minimum (tensor)
    gradient scale length was less than the current gyroradius.
    Similarly, count crossings from rL less than to greater than
    rL_thresh.

    Use exclude_before = 24367; exclude_after = None for those that escaped after t_s
    Use exclude_before = None ; exclude_after = 24367 for those that escape before t_s
    """

    #make a new data file to save 'collision' data
    collision_datafile_name = save_file_prefix+datetime.now().strftime('%Y-%m-%d_hr%Hm%M.npz')

    confined_iter = []
    collision_crossings = []
    large_rl_count = []
    collision_time_iter = []
    large_rl_time_iter = []
    file_list = []
    for filename in os.listdir(data_dir):
        if filename.endswith(file_suffix):
            file_list.append(filename)
    nfiles = len(file_list)
    start_time = time.time()

    for filecount, filename in enumerate(file_list):
        file_time = time.time()
        with h5py.File(os.path.join(data_dir,filename),'r') as file_obj:
            print('Reading data file {} of {}: '.format(filecount+1,nfiles) + filename)
            fkeys = list(file_obj.keys())
            for vindex in fkeys:
                #get trajectory and iterations
                iter = file_obj[vindex]['iter'][()]
                if exclude_before is not None:
                    if iter < exclude_before:
                        continue
                    else:
                        min_iter = exclude_before
                        max_iter = int(iter)

                if exclude_after is not None:
                    if iter > exclude_after:
                        continue
                    else:
                        min_iter = 0
                        max_iter = exclude_after

                r = file_obj[vindex]['r'][()][min_iter:max_iter:nskip,:]
                v = file_obj[vindex]['v'][()][min_iter:max_iter:nskip,:]
                B = file_obj[vindex]['B'][()][min_iter:max_iter:nskip,:]
                #calculate gyroradius and min gradient scale length
                rL = tft.get_rL(v,B)
                Lmin_traj = tft.get_scalar_along_saved_traj(r,gradLmin)
                #get number of iterations where scale length is smaller
                #    than the current gyroradius
                collision_indices = np.where(Lmin_traj < rL)[0]
                big_rl_indices = np.where(rL > rL_thresh)[0]

                Lmin_comparison = Lmin_traj - rL
                rL_comparison = rL - rL_thresh

                #get number of times orbit size crosses either gradient scale length
                # or large rL threshold
                collision_count = ((Lmin_comparison[:-1] * Lmin_comparison[1:]) < 0).sum()
                rl_count = ((rL_comparison[:-1] * rL_comparison[1:]) < 0).sum()

                #ratio of number of condition iterations to confined iterations tells us about confinement?
                confined_iter.append(iter)
                collision_crossings.append(collision_count)
                large_rl_count.append(rl_count)
                collision_time_iter.append(collision_indices.size)
                large_rl_time_iter.append(big_rl_indices.size)

        #save dataset as is and overwrite with more complete data next iteration of loop (next data file)
        np.savez(collision_datafile_name,collision_time_iter=np.array(collision_time_iter).squeeze(),
                large_rl_time_iter=np.array(large_rl_time_iter).squeeze(),
                collision_crossings=np.array(collision_crossings).squeeze(),
                large_rl_count=np.array(large_rl_count).squeeze(),
                confined_iter=np.array(confined_iter).squeeze(),
                total_iter=total_iter, rL_thresh=rL_thresh, data_dir=data_dir, latest_file=filename)
        print("File read time: {}\n".format(time.time()-file_time))

    print("Total run time: {}\n".format(time.time()-start_time))
    return np.array(collision_time_iter), np.array(large_rl_time_iter).squeeze(), np.array(confined_iter).squeeze(), total_iter, np.array(collision_crossings).squeeze(), np.array(large_rl_count).squeeze()



def concatenate_gradL_stats(escaped=True, write_combined_file=False,collision_datafile_name="collision_fractions_20241218_cyl270.npz"):
    if escaped:
        file_list = ["collision_fractions_27851396_2025-01-18_hr17m52.npz",
                    "collision_fractions_27912683_2025-01-18_hr16m00.npz",
                    "collision_fractions_27950944_2025-01-18_hr16m32.npz",
                    "collision_fractions_27950945_2025-01-18_hr16m59.npz",
                    "collision_fractions_27950946_2025-01-18_hr17m26.npz"]
    else:
        file_list = ["collision_fractions_27851396_confined_2025-01-19_hr13m15.npz",
                    "collision_fractions_27912683_confined_2025-01-19_hr23m18.npz",
                    "collision_fractions_27950944_confined_2025-01-20_hr13m39.npz",
                    "collision_fractions_27950945_confined_2025-01-20_hr14m11.npz",
                    "collision_fractions_27950946_confined_2025-01-20_hr14m39.npz"]

    collision_time_iter = np.empty(0)
    large_rl_time_iter = np.empty(0)
    collision_crossings = np.empty(0)
    large_rl_count = np.empty(0)
    confined_iter = np.empty(0)

    for file in file_list:
        data = np.load(file)
        collision_time_iter = np.hstack([collision_time_iter,data['collision_time_iter']])
        large_rl_time_iter = np.hstack([large_rl_time_iter,data['large_rl_time_iter']])
        collision_crossings = np.hstack([collision_crossings,data['collision_crossings']])
        large_rl_count = np.hstack([large_rl_count,data['large_rl_count']])
        confined_iter = np.hstack([confined_iter,data['confined_iter']])

    if write_combined_file:
        np.savez(collision_datafile_name,collision_time_iter=np.array(collision_time_iter).squeeze(),
            large_rl_time_iter=np.array(large_rl_time_iter).squeeze(),
            collision_crossings=np.array(collision_crossings).squeeze(),
            large_rl_count=np.array(large_rl_count).squeeze(),
            confined_iter=np.array(confined_iter).squeeze(),
            rL_thresh=data['rL_thresh'])

    return np.array(collision_time_iter), np.array(large_rl_time_iter).squeeze(),np.array(collision_crossings).squeeze(), np.array(large_rl_count).squeeze(), np.array(confined_iter).squeeze()


def plot_gradL_fraction(xvar, confined_iter, xlabel="Confinement Time ($10^3$ orbits)", ylabel="Time Fraction in Large Gradients", ax=None, log=False, vmin=1, vmax=100):
    """
    Make 2D histogram
    Set up by default for an xvar of Collision count/confined_iter but can use whatever.
    """

    if ax is None:
        fig = plt.figure(figsize=(3.37,2.5))
        fig.subplots_adjust(left=0.2,bottom=0.16,right=0.85,top=0.95,)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(pad=2, labelsize=8)
        ax.yaxis.set_tick_params(pad=2, labelsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel,y=0.48)
        fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
        yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        #ax.yaxis.set_ticks([0,0.05,0.1,0.15,0.2])
        #xticks = matplotlib.ticker.FormatStrFormatter('%.0e')
        #ax.xaxis.set_major_formatter(xticks)
        #ax.yaxis.set_ticks([0,0.05,0.1,0.15,0.2])
    else:
        fig = ax.get_figure()

    #mycmap = plt.cm.get_cmap("viridis").copy()
    #mycmap.set_under(mycmap(0),1)

    if log:
        #ax.hist2d(xvar,np.log10(np.clip(confined_iter,1,confined_iter.max())),bins=100)
        hist_params = ax.hist2d(np.log10(np.clip(xvar,1e-2,xvar.max())),np.log10(np.clip(confined_iter,1e-2,confined_iter.max())),bins=100,norm=colors.LogNorm(vmin=vmin,vmax=vmax))
    else:
        hist_params = ax.hist2d(xvar,confined_iter,bins=100, norm=colors.LogNorm(vmin=vmin,vmax=vmax))
        #ax.set_xlim(0,500)
        #ax.set_ylim(0,2e-1)


    ax.collections[-1].remove()
    h,xedges,yedges,qmesh = hist_params
    ax,cb,im = mytools.imview(np.log10(h.transpose()+1),x=xedges,y=yedges,cmap=plt.cm.viridis,ax=ax)
    im.set_clim(0,2)
    cb.ax.tick_params(labelsize=8)
    cb.set_ticks([0,1,2])
    cb.set_label('log10(number of particles)')
    ax.set_aspect('auto')
    ax.set_xlim(0,10)
    tt = ax.xaxis.get_offset_text()
    print(tt.get_text())
    #xexp.set_position(())
    ax.text(8.5e3,-0.09,tt.get_text())
    tt.set_visible(False)
    #qremesh.set_clim(1,100)
#    cb = fig.colorbar(mappable=im,label="Number of particles")
    return ax,cb

def plot_gradL_collision_number(xlabel="Gradient Collisions per 100 orbits", ylabel=r"Confinement Time ($\tau_0$)", ax=None, log=False, vmin=0.1, vmax=100):
    """
    Make 2D histogram
    """

    include_confined=False
    if include_confined:
        #confined
        collision_time_iter_conf, large_rl_time_iter_conf, collision_crossings_conf, large_rl_count_conf, confined_iter_conf = concatenate_gradL_stats(escaped=False)
        #escaped
        collision_time_iter, large_rl_time_iter, collision_crossings, large_rl_count, confined_iter = concatenate_gradL_stats(escaped=True)
        conf_time = np.hstack([confined_iter,confined_iter_conf])/100  #in fiducial orbits
        collision_count = np.hstack([collision_crossings,collision_crossings_conf])/conf_time  #collisions per 100 orbits
    else:
        collision_time_iter, large_rl_time_iter, collision_crossings, large_rl_count, confined_iter = concatenate_gradL_stats(escaped=True)
        conf_time = confined_iter/100  #in fiducial orbits
        collision_count = 100*collision_crossings/conf_time  #collisions per 100 orbits


    if ax is None:
        fig = plt.figure(figsize=(3.37,2.5))
        fig.subplots_adjust(left=0.2,bottom=0.16,right=0.85,top=0.95,)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(pad=2, labelsize=8)
        ax.yaxis.set_tick_params(pad=2, labelsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel,y=0.48)
        #fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
        #yticks = matplotlib.ticker.FormatStrFormatter(fmt)
        #ax.yaxis.set_major_formatter(yticks)
        #ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
        #ax.yaxis.set_ticks([0,0.05,0.1,0.15,0.2])
        #xticks = matplotlib.ticker.FormatStrFormatter('%.0e')
        #ax.xaxis.set_major_formatter(xticks)
        #ax.yaxis.set_ticks([0,0.05,0.1,0.15,0.2])
    else:
        fig = ax.get_figure()

    #mycmap = plt.cm.get_cmap("viridis").copy()
    #mycmap.set_under(mycmap(0),1)

    if log:
        #ax.hist2d(xvar,np.log10(np.clip(confined_iter,1,confined_iter.max())),bins=100)
        hist_params = ax.hist2d(np.log10(np.clip(collision_count,1,collision_count.max())),np.log10(np.clip(conf_time,1,conf_time.max())),bins=100,norm=colors.LogNorm(vmin=vmin,vmax=vmax))
    else:
        hist_params = ax.hist2d(collision_count,conf_time,bins=100, norm=colors.LogNorm(vmin=vmin,vmax=vmax))
        #ax.set_xlim(0,500)
        #ax.set_ylim(0,2e-1)


    ax.collections[-1].remove()
    h,xedges,yedges,qmesh = hist_params
    ax,cb,im = mytools.imview(np.log10(h.transpose()+0.9),x=xedges,y=yedges,cmap=plt.cm.viridis,ax=ax)
    im.set_clim(0,2)
    cb.ax.tick_params(labelsize=8)
    cb.set_ticks([0,1,2])
    cb.set_label('log10(number of particles)')
    ax.set_aspect('auto')
    ax.set_xlim(0,20) #per 100 orbits
    ax.set_ylim(0,1e4)
    return ax,cb

def make_gradL_histograms(early=24367,ax=None,num_bins=100,xlim=[0,100],ylim=[0,2.5],ntot=27000,semilogy=False,title=False,legend=True,stacked_bar=False,plot_total=False,norm_per_bin=False):
    """
    early, escaping, confined histograms for fraction of time spend in high gradient regions
    """

    if ax is None:
        fig = plt.figure(figsize=(3.37,2.5))
        fig.subplots_adjust(left=0.21,bottom=0.16,right=0.93,top=0.97,)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(pad=2, labelsize=8)
        ax.yaxis.set_tick_params(pad=2, labelsize=8)

    #confined
    collision_time_iter, large_rl_time_iter, collision_crossings, large_rl_count, confined_iter = concatenate_gradL_stats(escaped=False)
    frac_conf = 100*collision_time_iter/confined_iter

    #escaped
    collision_time_iter, large_rl_time_iter, collision_crossings, large_rl_count, confined_iter = concatenate_gradL_stats(escaped=True)
    early_inds = confined_iter < early
    frac_early = 100*collision_time_iter[early_inds]/confined_iter[early_inds]
    esc_inds = confined_iter > early
    frac_esc = 100*collision_time_iter[esc_inds]/confined_iter[esc_inds]

    num_early,bins_early = np.histogram(frac_early,bins=num_bins,range=xlim)
    num_esc,bins_esc = np.histogram(frac_esc,bins=bins_early,range=xlim)
    num_conf,bins_conf = np.histogram(frac_conf,bins=bins_early,range=xlim)


    #max_mu = np.max([bins[-1],bins_esc[-1]]))
    bar_width = (bins_conf[-1] - bins_conf[0]) / num_bins

    #plot as percent, so calculate fraction of total in each bin * 100

    if semilogy:
        stacking_order = [1, 2, 3]
    else:
        stacking_order = [3, 2, 1]

    if norm_per_bin:
        total_at_each_gradL = num_early+num_esc+num_conf
        ntot=total_at_each_gradL
        ylim=(0,100)

    if stacked_bar:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Esc. Early', width=bar_width, log=semilogy, zorder=stacking_order[0])
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Esc. Later', width=bar_width, log=semilogy, zorder=stacking_order[1],bottom=100*num_early/ntot)
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, zorder=stacking_order[2],bottom=(100*num_esc/ntot + 100*num_early/ntot))
    else:
        ax.bar(bins_early[:-1], 100*num_early/ntot, align='edge', alpha=0.4, color='0.4', label='Esc. Early', width=bar_width, log=semilogy, zorder=stacking_order[0])
        ax.bar(bins_esc[:-1], 100*num_esc/ntot, align='edge', alpha=0.4, color='r', label='Esc. Later', width=bar_width, log=semilogy, zorder=stacking_order[1],)
        ax.bar(bins_conf[:-1], 100*num_conf/ntot, align='edge', alpha=0.4, color='b', label='Confined', width=bar_width, log=semilogy, zorder=stacking_order[2],)


    if plot_total:
        total_at_each_angle = num_early+num_esc+num_conf
        ax.plot(bins_early[:-1],100*total_at_each_angle/27000,'k--',label='Total')


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    xticks = matplotlib.ticker.FormatStrFormatter('%.0f%%')
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Fraction of Time in Large Gradients')
    ax.set_ylabel('Fraction of Particles')
    if title:
        ax.set_title('test title',y=1.03)

    if legend:
        ax.legend(bbox_to_anchor=(1,1),loc='upper right',fontsize=8,framealpha=1)
    ax.grid()




def get_small_scalelength_crossings(gradLmin, data_dir, smooth=None, file_suffix='escaped.h5', save_file_prefix='collision_counts_', nskip=100, rL_thresh=5, total_iter = 1000000):
    """
    Recover number of iterations completed before escape
    and the number of times during the run that the particle
    crossed into or out of a region where the minimum (tensor)
    gradient scale length was less than the current gyroradius.
    Also returns count of crossings from rL less than to greater than
    rL_thresh.
    """

    #make a new data file to save 'collision' data
    collision_datafile_name = save_file_prefix+datetime.now().strftime('%Y-%m-%d_hr%Hm%M.npz')

    confined_iter = []
    collision_crossings = []
    large_rl_count = []
    file_list = []
    for filename in os.listdir(data_dir):
        if filename.endswith(file_suffix):
            file_list.append(filename)
    nfiles = len(file_list)
    start_time = time.time()

    for filecount, filename in enumerate(file_list):
        file_time = time.time()
        with h5py.File(os.path.join(data_dir,filename),'r') as file_obj:
            print('Reading data file {} of {}: '.format(filecount+1,nfiles) + filename)
            fkeys = list(file_obj.keys())
            for vindex in fkeys:
                #get trajectory and iterations
                r = file_obj[vindex]['r'][()]
                v = file_obj[vindex]['v'][()]
                B = file_obj[vindex]['B'][()]
                iter = file_obj[vindex]['iter'][()]
                #calculate gyroradius and min gradient scale length
                rL = tft.get_rL(v,B)
                Lmin_traj = tft.get_scalar_along_saved_traj(r,gradLmin)

                if smooth is not None:
                    rL = mytools.smooth(rL,window_len=smooth)
                    Lmin_traj = mytools.smooth(Lmin_traj,window_len=smooth)

                Lmin_comparison = Lmin_traj - rL
                rL_comparison = rL - rL_thresh

                #get number of terations where scale length is smaller
                #    than the current gyroradius
                collision_count = ((Lmin_comparison[:-1] * Lmin_comparison[1:]) < 0).sum()
                rl_count = ((rL_comparison[:-1] * rL_comparison[1:]) < 0).sum()

                #ratio of number of condition iterations to confined iterations tells us about confinement?
                confined_iter.append(iter)
                collision_crossings.append(collision_count)
                large_rl_count.append(rl_count)

        #save dataset as is and overwrite with more complete data next iteration of loop (next data file)
        np.savez(collision_datafile_name,collision_crossings=np.array(collision_crossings).squeeze(), large_rl_count=np.array(large_rl_count).squeeze(), confined_iter=np.array(confined_iter).squeeze(), total_iter=total_iter, rL_thresh=rL_thresh, data_dir=data_dir, smooth=smooth)
        print("File read time: {}\n".format(time.time()-file_time))

    print("Total run time: {}\n".format(time.time()-start_time))
    return np.array(collision_crossings), np.array(large_rl_count).squeeze(), np.array(confined_iter).squeeze(), total_iter


def get_merged_trajectory_gradient_collisions(data1, data2, data3):
    """
    """
    pass


def plot_gradient_collision_fractions(iter_confined, collision_count, total_iter=300000, fit_degree=1, title='Adjust Title to Match Run Data'):
    """
    """

    sort_inds = iter_confined.argsort()

    x_sorted = collision_count[sort_inds]/total_iter
    y_sorted = iter_confined[sort_inds]/total_iter

    fit_coeffs = np.polyfit(x_sorted, y_sorted, fit_degree)
    fit_func = np.poly1d(fit_coeffs)

    fig = plt.gcf()
    plt.clf()
    ax = fig.add_subplot(111)
    first_half_lost = x_sorted<0.5
    ax.plot(x_sorted[first_half_lost], y_sorted[first_half_lost], 'or', alpha=0.1)
    ax.plot(x_sorted, y_sorted, 'ob', alpha=0.1)
    ax.plot(x_sorted, fit_func(x_sorted), linewidth=2)
    ax.set_xlabel("Fraction of Trajectory Experiencing Gradient Collisions")
    ax.set_ylabel("Confinement Time (normalized to total simulation time)")
    ax.set_title(title, y=1.02)

    ax.set_ylim(0,1)
    ax.set_xlim(0,1)

    return ax

def plot_Lgrad_histogram(cached_B_tuple,num_bins=50,semilogy=False,xlim=(0,50),ylim=(0,8)):
    """
    Constructs histogram of rho/L_grad values for everywhere in the cylinder.
    """
    L_grad = tft.get_tensor_scale_lengths(cached_B_tuple,minimum_only=True)
    x, y, z, bx, by, bz = cached_B_tuple
    rr = np.sqrt(x**2 + y**2)
    in_cylinder = (rr<1.0) & (z >0) & (z<10)
    #ratio = 1/L_grad[in_cylinder]

    ntot = in_cylinder.sum()
    num,bins = np.histogram(L_grad[in_cylinder],bins=num_bins,range=xlim)

    fig = plt.figure(figsize=(5,3))
    fig.subplots_adjust(left=0.12,bottom=0.15,right=0.95,top=0.89)
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=8)

    bar_width = (bins[-1] - bins[0]) / num_bins

    ax.bar(bins[:-1], 100*num/ntot, align='edge', alpha=0.4, color='0.4', label='', width=bar_width, log=semilogy)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fmt = '%.1f%%' # Format you want the ticks, e.g. '40%'
    yticks = matplotlib.ticker.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_xlabel('Magnetic Field Gradient Scale Length (fiducial orbit radius)')
    ax.set_ylabel('Fraction of Particles')

    ax.grid()


def get_helicities_vs_confinement_time(single_pos_file):

    confined_iter = []
    kinetic_helicity = []
    cross_helicity = []
    #with h5py.File(os.path.join(data_dir,filename),'r') as file_obj:
    with h5py.File(single_pos_file,'r') as file_obj:
        #print('Reading data file {} of {}: '.format(filecount+1,nfiles) + filename)
        fkeys = list(file_obj.keys())
        for vindex in fkeys:
            #get trajectory and iterations
            r = file_obj[vindex]['r'][()]
            v = file_obj[vindex]['v'][()]
            B = file_obj[vindex]['B'][()]
            iter = file_obj[vindex]['iter'][()]
            #calculate vorticity for kinetic helicity
            vorticity = tft.get_fluid_vorticity_along_r(r,v)
            #cross_helicity.append((v*B).sum(axis=-1))
            confined_iter.append(iter)
            kinetic_helicity.append((v*vorticity).sum())
            cross_helicity.append((v*B).sum())

    return confined_iter, kinetic_helicity, cross_helicity



def get_pitch_angle(v,B):
    """
    Takes arrays of v and B, calculates pitch angle element-wise.
    """
    vmag = np.sqrt((v*v).sum(axis=1))
    Bmag = np.sqrt((B*B).sum(axis=1))

    return np.arccos((v*B).sum(axis=1)/(vmag*Bmag))


# Check to see if this file is being executed as the "Main" python
# script instead of being used as a module by some other python script
# This allows us to use the module which ever way we want.
if __name__ == '__main__':

    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("directory")
    parser.add_argument("--read-escaped", help="switch to analyzing escaped trajectories",action="store_true")
    parser.add_argument("--outfile", help="specify output file name")

    # read arguments from the command line
    args = parser.parse_args()

    ri,rf,vi,Bi,max_iter,is_confined = read_single_position_files(args.directory,                                                     read_escaped=args.read_escaped, save_output=False)

    if args.outfile:
        outfile = args.outfile
    else:
        outfile = "particle_summary.npz"


    print(args)
    np.savez(outfile,initial_positions=ri,final_positions=rf,initial_velocities=vi,initial_fields=Bi,max_iterations=max_iter,is_confined=is_confined)
