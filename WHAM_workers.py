import particle_sieve as ps
import taylor_field_tools as tft
import WHAMField
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import maxwell
from scipy.stats import uniform_direction
from scipy.ndimage import gaussian_filter1d
import os
import h5py
import matplotlib.pyplot as plt
import time
import orbit_statistics
import tracemalloc
import pandas as pd

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
    
    tracemalloc.start()
    
    try:
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
    except MemoryError:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("MemoryError in worker. Top allocations:")
        for stat in top_stats[:20]:
            print(stat)
        raise MemoryError


def RunGrid(norbits, nvel, vertices, dt=0.1, m=1, q=1, T=1, B0=1, scale=1, 
                    shaper=(0,0.4), shapez=(-1,1), filename='data//WHAMTest//Storage_test//'):
    
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    shaper *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    shapez *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    bufferr = shaper[1] / 10
    bufferz = shapez[1] / 10
    rr = np.repeat(np.linspace(shaper[0]+bufferr, shaper[1]-bufferr, 5, endpoint=True), nvel)
    zz = np.linspace(shapez[0]+bufferz, shapez[1]-bufferz, 20, endpoint=True)
    
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
    
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(new_run_position, *args): args for args in all_args}
        for future in as_completed(futures):
            try:
                particle, filename, v = future.result()  # this re-raises the actual exception from the worker
                ps.write_single_position_data(particle,filename,f"v{v:03.3f}",write_mode='a')
                print("Finished", filename, v)
                del particle
            except Exception as e:
                print(f"Worker failed with: {type(e).__name__}: {e}")
                if e == MemoryError:
                    
                    break
                continue
           
        
        
def plot_z_vs_t(file_path, savedir=None):
    file = file_path.split("/")[-1]
    with h5py.File(file_path, mode='r') as ff:
        for i, v in enumerate(ff.keys()):
            plt.plot(list(range(len(ff[v]["r"]))), ff[v]['r'][:,2])
            plt.xlabel("Iteration")
            plt.ylabel("Z-position")
            plt.title(f"File {file}, Particle number {i}")
            if savedir != None:
                plt.savefig(savedir)
            plt.show()
            plt.close()
            
def plot_trajectory(file_path, savedir=None):
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
            if savedir != None:
                plt.savefig(savedir)
            plt.show()
            plt.close()

def plot_zs_vs_t(directory, confined=True, savedir=None):
    files = os.listdir(directory)
    for file in files:
        if (file.endswith("_confined.h5") and confined) or (file.endswith("_escaped.h5") and not confined):
            with h5py.File(os.path.join(directory,file), mode='r') as ff:
                for i, v in enumerate(ff.keys()):
                    plt.plot(list(range(len(ff[v]["r"]))), ff[v]['r'][:,2])
    plt.xlabel("Iteration")
    plt.ylabel("Z-position")
    plt.title("Particle positions vs time")
    if savedir != None:
        plt.savefig(savedir)
    plt.show()
    plt.close()

def plot_trajectories(directory, confined=True, savedir=None):
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
    if savedir != None:
        plt.savefig(savedir)
    plt.show()
    plt.close(fig)

def get_fraction_lost(directory):
    files = os.listdir(directory)
    confined = 0
    escaped = 0
    for file in files:
        if file.endswith('_confined.h5'):
            with h5py.File(os.path.join(directory, file), mode='r') as ff:
                for _ in ff.keys():
                    confined += 1
        elif file.endswith('_escaped.h5'):
            with h5py.File(os.path.join(directory, file), mode='r') as ff:
                for _ in ff.keys():
                    escaped += 1
    frac_confined = confined / (confined + escaped)
    print(f"Total confined: {confined}")
    print(f"Total escaped: {escaped}")
    print(f"Fraction confined: {frac_confined}")

def confined_in_vperp_vpar_space(directory, savedir=None):
    files = os.listdir(directory)
    confined = [[],[]]
    escaped = [[],[]]
    for file in files:
        if file.endswith('_confined.h5'):
            with h5py.File(os.path.join(directory, file), mode='r') as ff:
                for i, v in enumerate(ff.keys()):
                    v0 = ff[v]['v'][0]
                    vpar = np.abs(v0[2])
                    vperp = np.sqrt(v0[0]**2 + v0[1]**2)
                    confined[0].append(vpar)
                    confined[1].append(vperp)
        elif file.endswith('_escaped.h5'):
            with h5py.File(os.path.join(directory, file), mode='r') as ff:
                for i, v in enumerate(ff.keys()):
                    v0 = ff[v]['v'][0]
                    vpar = np.abs(v0[2])
                    vperp = np.sqrt(v0[0]**2 + v0[1]**2)
                    escaped[0].append(vpar)
                    escaped[1].append(vperp)
                    
    plt.plot(confined[1], confined[0], 'o', label="Confined", color='blue')
    plt.plot(escaped[1], escaped[0], 'x', label="Escaped", color='red')
    plt.ylabel('Normalized Parallel Velocity')
    plt.xlabel('Normalized Perpendicular Velocity')
    plt.title('Confinement by Position in Velocity Space')
    plt.legend()
    if savedir != None:
        plt.savefig(savedir)
    plt.show()
    plt.close()

def confinement_over_time(directory, smooth=True, savedir=None):
    _, _, _, _, tfinal, _ = orbit_statistics.read_single_position_files(directory, read_escaped=False, save_output=False)
    _, _, _, _, tlost, _ = orbit_statistics.read_single_position_files(directory, read_escaped=True, save_output=False)
    
    tlost = np.sort(np.array(tlost))
    survival = 1 - np.arange(1, len(tlost) + 1) / (len(tlost) + len(tfinal))
    
    times = np.concatenate([[0], tlost, [tfinal[0]]])
    survival = np.concatenate([[1], survival, [survival[-1]]])

    t_fine = np.linspace(0, tfinal[0], 10000)
    survival_fine = np.interp(t_fine, times, survival)
    
    survival_smooth = gaussian_filter1d(survival_fine, sigma=100)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    if smooth:
        ax.plot(t_fine, survival_smooth, color='steelblue', linewidth=2.5, label='Confined fraction')
    else:
        ax.plot(times, survival, color='steelblue', linewidth=2.5, label='Confined fraction')
     
    ax.fill_between(t_fine, survival_smooth, alpha=0.15, color='steelblue')
    
    ax.set_xlabel('Time (normalized)', fontsize=13)
    ax.set_ylabel('Proportion of Particles Confined', fontsize=13)
    ax.set_title("Particle confinement over time", fontsize=14)
    ax.set_xlim(0, tfinal[0])
    ax.set_ylim(0, 1.02)
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11)
    plt.tight_layout()
    if savedir != None:
        plt.savefig(savedir)
    plt.show()
        
def plot_confinement_with_fieldlines(directory, bFunc, scale=1/0.000102, savedir=None):
    conf_pos, _, _, _, _, _ = orbit_statistics.read_single_position_files(directory, read_escaped=False, save_output=False)
    esc_pos, _, _, _, _, _ = orbit_statistics.read_single_position_files(directory, read_escaped=True, save_output=False)
    
    unique_pos = np.unique(np.concatenate([conf_pos, esc_pos]), axis=0)
    
    df = pd.DataFrame(columns=['r', 'z', 'conf'])
    for pos in unique_pos:
        r = np.sqrt(pos[0]**2 + pos[1]**2)
        n_conf = np.count_nonzero(np.count_nonzero(conf_pos == pos, axis=1) == 3)
        n_esc = np.count_nonzero(np.count_nonzero(esc_pos == pos, axis=1) == 3)
        temp_df = pd.DataFrame({'r': [r], 'z': [pos[2]], 'conf': [n_conf / (n_conf + n_esc)]})
        df = pd.concat([df, temp_df])
    
    rr = np.linspace(0, 0.2 * scale, 100)
    zz = np.linspace(-1 * scale, 1 * scale, 1000)
    Z, R = np.meshgrid(zz, rr)
    BR = np.zeros_like(R)
    BZ = np.zeros_like(Z)
    
    for i in range(len(rr)-1):
        for j in range(len(zz)-1):
            B = bFunc([rr[i], 0, zz[j]])
            BR[i, j] = B[0]
            BZ[i, j] = B[2]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Confinement scatter plot ---
    sc = ax.scatter(
        df['z'], df['r'],
        c=df['conf'],
        cmap='plasma_r',
        vmin=0, vmax=1,
        s=50,           # marker size, adjust to taste
        zorder=2        # draw points on top of field lines
    )
    cbar = fig.colorbar(sc, ax=ax, label='Particles Confined (%)', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # --- Magnetic field lines ---
    B_magnitude = np.sqrt(BR**2 + BZ**2)
    BR_norm = BR / (B_magnitude + 1e-10)
    BZ_norm = BZ / (B_magnitude + 1e-10)
    
    ax.streamplot(
        zz, rr,
        BZ_norm, BR_norm,
        color='steelblue',
        linewidth=1.0,
        density=1,
        arrowsize=1.0,
        zorder=1        # draw field lines underneath the points
    )
    
    # ----------------------------------------------------------------
    # 4. Formatting
    # ----------------------------------------------------------------
    ax.set_xlabel('z (ion gyroradii)', fontsize=13)
    ax.set_ylabel('r (ion gyroradii)', fontsize=13)
    ax.set_title('Particle Confinement by Position with Magnetic Field Lines', fontsize=14)
    ax.set_xlim(zz.min(), zz.max())
    ax.set_ylim(rr.min(), rr.max())
    ax.tick_params(labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig, ax


