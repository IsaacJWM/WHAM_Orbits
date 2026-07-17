import particle_sieve as ps
import WHAMField
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import maxwell
from scipy.stats import uniform_direction
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
import os
import h5py
import matplotlib.pyplot as plt
import orbit_statistics
import pandas as pd

sys.path.insert(0, "./classes")

import Particlev03 as pt


def run_particle_in_grid(position,bFunc,vertices,norbits=100,dt=0.01,seed=0):
    """
    Takes a particle's starting position, generates a randoom velocity, and runs the particle.
    """
    rng = np.random.default_rng(seed)
    
    vx = ps.select_velocities(1, rng)
    vy = ps.select_velocities(1, rng)
    vz = ps.select_velocities(1, rng)
    print(vx, vy, vz)
    p1 = pt.particle(position, np.array([vx[0], vy[0], vz[0]]), dt, int(norbits * 2 * np.pi / dt), silent=True)
    p1.set_boundaries(vertices=vertices)
    p1.step(bFunc)
    
    return p1


def RunGrid(norbits, nvel, vertices, dt=0.1, m=1, q=1, T=1, B0=1, scale=1, 
                    shaper=(0,0.4), shapez=(-1,1), filepath='data//WHAMTest//'):
    
    nr = 8
    nz = 20    
    
    field_data = WHAMField.WHAMField(m=m, q=q, B0=B0, T=T, scale=scale)
    
    shaper *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    shapez *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    bufferr = shaper[1] / 10
    bufferz = shapez[1] / 10
    rr = np.repeat(np.linspace(shaper[0]+bufferr, shaper[1]-bufferr, nr, endpoint=True), nvel)
    zz = np.linspace(shapez[0]+bufferz, shapez[1]-bufferz, nz, endpoint=True)
    ss = np.random.SeedSequence()
    seeds = ss.spawn(len(rr) * len(zz))
    
    vertices *= (scale/0.000102) *np.sqrt(m*T) / (q*B0)
    
    #for i, r in enumerate(rr):
    #    for j, z in enumerate(zz):
    #        run_particle_in_grid([0,r,z], field_data.field, vertices, norbits, dt, seeds[len(rr)*i+j])
    #        print("Concluded particle", r, z)
    
    args_unseeded = [
        (np.array([0, rloc, zloc]), field_data.field, vertices, norbits, dt)
        for zloc in zz
        for rloc in rr
        ]
    
    all_args = [(*args, s) for args, s in zip(args_unseeded, seeds)]
    
    data = pd.DataFrame(index=range(nr * nz * nvel), columns=["x0", "v0", "xf", "yf", "iter", "conf", "success"])
    max_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 16))
    zs = np.array([0,0,0])
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_particle_in_grid, *args): args for args in all_args}
        count = 0
        for future in as_completed(futures):
            try:
                p = future.result()  # this re-raises the actual exception from the worker
                data.loc[count] = [p.r0, p.v0, p.r, p.v, p.iter, p.outOfBounds == False, p.success]
                count += 1
                print(f"Finished count: {count}. Iterations: {p.iter}. Time per iteration: {p.iter_time}")
            except Exception as e:
                print(f"Particle #{count} failed with: {type(e).__name__}: {e}")
                data.loc[count] = [zs, zs, zs, zs, 0, False, False]
                count += 1
    
    data.to_pickle(os.path.join(filepath, "output.pkl"))

def read_data(fname):
    df = pd.read_pickle(fname)
    df = df[df["success"]]
    conf = df[df["conf"]]
    esc = df[df["conf"] == False]
    return conf, esc

def get_fraction_lost(conf, esc):
    confined = len(conf)
    escaped = len(esc)
    total = escaped + confined
    
    if total == 0:
        print("The dataframe passed was empty")
    else:
        frac_confined = confined / (confined + escaped)
        print(f"Total: {total}")
        print(f"Total confined: {confined}")
        print(f"Total escaped: {escaped}")
        print(f"Fraction confined: {frac_confined}")

def confined_in_vperp_vpar_space(conf, esc, savedir=None):
    if len(conf) != 0:
        conf_par = np.abs(np.stack(conf["v0"])[:,2])
        conf_perp = np.sqrt(np.stack(conf["v0"])[:,0]**2 + np.stack(conf["v0"])[:,1]**2)
        plt.plot(conf_perp, conf_par, '.', label="Confined", color='blue')
    if len(esc) != 0:
        esc_par = np.abs(np.stack(esc["v0"])[:,2])
        esc_perp = np.sqrt(np.stack(esc["v0"])[:,0]**2 + np.stack(esc["v0"])[:,1]**2)
        plt.plot(esc_perp, esc_par, 'x', label="Escaped", color='red')
    plt.ylabel('Normalized Parallel Velocity')
    plt.xlabel('Normalized Perpendicular Velocity')
    plt.title('Confinement by Position in Velocity Space')
    plt.legend()
    if savedir != None:
        plt.savefig(os.path.join(savedir, "Vperp_Vpar.png"))
        plt.close()
        return
    plt.show()
    plt.close()

def confinement_over_time(conf, esc, smooth=True, savedir=None):
    if len(esc) == 0:
        print("Error, no particles escaped")
        return
    
    tfinal = np.stack(conf["iter"])
    tlost = np.stack(esc["iter"])
    
    tlost = np.sort(tlost)
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
        plt.savefig(os.path.join(savedir, "Confinement_over_time.png"))
        plt.close()
        return
    plt.show()
    plt.close()
        
def plot_confinement_with_fieldlines(conf, esc, bFunc, scale=1/0.000102, savedir=None):
    if len(conf) != 0:
        conf_pos = np.stack(conf["x0"])
    else:
        conf_pos = np.array([])
    if len(esc) != 0:
        esc_pos = np.stack(esc["x0"])
    else:
        esc_pos = np.array([])
    
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
        cmap='plasma',
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
    if savedir != None:
        plt.savefig(os.path.join(savedir, "Confinement_by_position.png"))
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)
    return fig, ax


def plot_confined_by_pitch_angle(conf, esc, savedir=None):
    n_bins = 50
    bins = np.linspace(-np.pi/2, np.pi/2, n_bins + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if len(conf) != 0:
        ivel_conf = np.stack(conf["v0"])
        pa_conf = np.arctan(np.sqrt(ivel_conf[:,0]**2 + ivel_conf[:,1]**2) / ivel_conf[:,2])
        ax.hist(pa_conf, bins=bins, alpha=0.6, color='steelblue', 
                label=f'Confined (n={len(pa_conf)})', edgecolor='white', linewidth=0.5)
    if len(esc) != 0:
        ivel_esc = np.stack(esc["v0"])
        pa_esc = np.arctan(np.sqrt(ivel_esc[:,0]**2 + ivel_esc[:,1]**2) / ivel_esc[:,2])
        ax.hist(pa_esc, bins=bins, alpha=0.6, color='coral',
            label=f'Escaped (n={len(pa_esc)})', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Pitch Angle (radians)', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Pitch Angle Distribution: Confined vs Escaped', fontsize=14)
    ax.legend(fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    if savedir != None:
        plt.savefig(os.path.join(savedir, "Pitch_angle.png"))
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)

def plot_escaped_positions_2d(esc, boundary, bFunc, scale, savedir=None):
    """
    escape_positions: array of shape (N, 3) with x,y,z positions where particles escaped
    boundary: array of boundary positions in r-z coordinates
    """
    esc_xf = np.stack(esc['xf'])
    r = np.sqrt(esc_xf[:,0]**2 + esc_xf[:,1]**2)
    z = esc_xf[:,2]
    esc_x0 = np.stack(esc['x0'])
    r0 = esc_x0[:,1]
    z0 = esc_x0[:,2]
    
    rr = np.linspace(0, 0.25 * scale, 100)
    zz = np.linspace(-1.25 * scale, 1.25 * scale, 1000)
    Z, R = np.meshgrid(zz, rr)
    BR = np.zeros_like(R)
    BZ = np.zeros_like(Z)
    
    for i in range(len(rr)-1):
        for j in range(len(zz)-1):
            B = bFunc([rr[i], 0, zz[j]])
            BR[i, j] = B[0]
            BZ[i, j] = B[2]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(boundary[:,1], boundary[:,0], color='gray', linewidth=5, zorder=1, label='Confinement boundary')
    ax.scatter(z, r, marker='x', color='red', zorder=2, label=f'Escape positions (n={len(esc)})')
    ax.scatter(z0, r0, marker='o', color='green', label='Starting positions')
    
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
        zorder=1,        # draw field lines underneath the points
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
    
    
    plt.xlabel("z (ion gyroradii)")
    plt.ylabel("r (ion gyroradii)")
    plt.legend()
    plt.title("Particle Escape Positions")
    if savedir != None:
        plt.savefig(os.path.join(savedir, "Escape_positions_2d.png"))
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)


def plot_escaped_positions_3d(esc, boundary, savedir=None):
    """
    escape_positions: array of shape (N, 3) with x,y,z positions where particles escaped
    boundary: array of boundary positions in r-z coordinates
    """

    # ----------------------------------------------------------------
    # 1. Revolve the 2D boundary curve around the z-axis
    # ----------------------------------------------------------------
    boundary_r = boundary[:,0]
    boundary_z = boundary[:,1]
    theta = np.linspace(0, 2 * np.pi, 100)

    # Meshgrid over boundary points and azimuthal angles
    R, Theta = np.meshgrid(boundary_r, theta)
    Z, _ = np.meshgrid(boundary_z, theta)

    # Convert to Cartesian
    X_surf = R * np.cos(Theta)
    Y_surf = R * np.sin(Theta)
    Z_surf = Z

    # ----------------------------------------------------------------
    # 2. Build the plot
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')

    # --- Boundary surface ---
    ax.plot_surface(
        X_surf, Z_surf, Y_surf,   # z-axis up convention: pass y as third arg
        alpha=0.15,                # very transparent so escape points are visible
        color='steelblue',
        edgecolor='none',
        zorder=1
    )
    
    # --- Escape positions ---
    x_esc = np.stack(esc['xf'])[:, 0]
    y_esc = np.stack(esc['xf'])[:, 1]
    z_esc = np.stack(esc['xf'])[:, 2]

    ax.scatter(
        x_esc, z_esc, y_esc,      # y-axis up convention
        c='coral',
        s=20,
        alpha=0.8,
        zorder=3,
        label=f'Escape positions (n={len(esc)})'
    )
    
    # ----------------------------------------------------------------
    # 3. Formatting
    # ----------------------------------------------------------------
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_zlabel('y', fontsize=12)
    ax.set_title('Particle Escape Positions', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    if savedir != None:
        plt.savefig(os.path.join(savedir, "Escape_Positions_3d.png"))
        plt.close(fig)
        return
    plt.show()
    plt.close(fig)
    return fig, ax


def plot_3d_fieldlines(bFunc, scale=1/0.000102):
    """
    bFunc: function((x, y, z)) -> (Bx, By, Bz)
    """
    
    figsize=(10,8)
    dt=0.01 * scale
    t_max=3 * scale
    
    seed_points = np.array([
        [r * np.cos(t), r * np.sin(t), -3]
        for r in [0.05, 0.1, 0.15, 0.2]
        for t in np.linspace(0, 2*np.pi, 8, endpoint=False)
        ]) * scale
    
    def field_ode(t, pos):
        Bx, By, Bz = bFunc(pos)
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2) + 1e-10
        # normalize so integration speed doesn't depend on field strength
        return [Bx/B_mag, By/B_mag, Bz/B_mag]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, int(t_max/dt))

    for seed in seed_points:
        # integrate forward
        sol_fwd = solve_ivp(
            field_ode, t_span, seed,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6, atol=1e-8
        )
        # integrate backward (reverse field direction)
        sol_bwd = solve_ivp(
            lambda t, pos: [-v for v in field_ode(t, pos)],
            t_span, seed,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-6, atol=1e-8
        )

        # combine backward (reversed) and forward solutions
        x = np.concatenate([sol_bwd.y[0][::-1], sol_fwd.y[0]])
        y = np.concatenate([sol_bwd.y[1][::-1], sol_fwd.y[1]])
        z = np.concatenate([sol_bwd.y[2][::-1], sol_fwd.y[2]])

        # color by field strength along the line
        B_mag = np.array([
            np.sqrt(sum(b**2 for b in bFunc(np.array([x[i], y[i], z[i]]))))
            for i in range(len(x))
        ])

        # plot as a colored line using scatter for color gradient
        ax.plot(x, z, y, alpha=0.7, linewidth=120)  # y-axis up

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title('3D Magnetic Field Lines')
    plt.tight_layout()
    plt.show()
    return fig, ax