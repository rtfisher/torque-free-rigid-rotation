#!/usr/bin/env python3
"""
Animate torque-free rigid body rotation with unequal principal moments of
inertia.

-rtf, 110325, written with the assistance of Claude Code.

We produce TWO synchronized 3D views by default, or THREE with --poinsot flag:

1. "Inertial Frame":
   - The box tumbles in space.
   - ω (red) and L (green) drawn in the inertial (space) frame.
   - A trail of recent ω-space.
   - A red marker dot painted on the body so you can watch material rotation.

2. "Body-Centered Frame":
   - The body is frozen (we ride with the body).
   - ω_body(t) and L_body(t) are drawn relative to that body.
   - A trail of recent ω_body.
   - The same red dot in body coordinates.

3. "Poinsot Ellipsoids" (optional, shown with --poinsot flag):
   - Shows ω in BODY FRAME where inertia tensor is diagonal
   - Two ellipsoids in body-frame ω-space representing the conserved quantities:
     * Momentum ellipsoid (green): I₁²ω₁² + I₂²ω₂² + I₃²ω₃² = L²
     * Energy ellipsoid (orange, transparent): I₁ω₁² + I₂ω₂² + I₃ω₃² = 2E
   - Their intersection curve (yellow) along which ω_body moves.
   - The trajectory of ω_body(t) (red trail) showing the Poinsot construction.
   - The ellipsoids are FIXED in the body frame.

Key physics:
- We solve pure torque-free Euler rigid body dynamics.
- No external torque.
- No damping / dissipation.
- L in inertial space stays fixed (up to numerical error).
- Energy and |L| are conserved (up to numerical error).
- The intermediate axis instability is visible when spinning mostly about I2.

Command-line features:
- --stable1 / --unstable / --stable3 presets for initial ω.
- --w0 to fully override initial ω.
- --dims to set box dimensions (drawn box, and where the red dot lives).
- --I to set principal moments of inertia [I1, I2, I3].
- --dotpos X Y Z chooses where on the body the red dot sits,
  in FRACTIONS of HALF the box lengths.
  Default is 0 1 0 (center of +y face).

- --fps, --tmax for temporal resolution and total duration.
- --outfile to save MP4 (requires ffmpeg).
- --vecscale to scale ω and L arrows.
- --traillen to control how long the ω trails are.
- --sliders to enable interactive sliders for dynamically adjusting initial angular velocity.

Dependencies: numpy, scipy, matplotlib
"""

import argparse
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.integrate import solve_ivp


# ------------------------------------------------------------
# Physics (pure torque-free rigid body)
# ------------------------------------------------------------

def euler_rhs(omega_body, I):
    """
    Classic torque-free Euler equations in the BODY frame.

    Let I = (I1, I2, I3) be the principal moments and
    ω_body = (ω1, ω2, ω3) be the angular velocity components in that frame.

    Equations:
      dω1/dt = ((I2 - I3)/I1) * ω2 * ω3
      dω2/dt = ((I3 - I1)/I2) * ω3 * ω1
      dω3/dt = ((I1 - I2)/I3) * ω1 * ω2
    """
    I1, I2, I3 = I
    w1, w2, w3 = omega_body

    w1dot = ((I2 - I3) / I1) * w2 * w3
    w2dot = ((I3 - I1) / I2) * w3 * w1
    w3dot = ((I1 - I2) / I3) * w1 * w2

    return np.array([w1dot, w2dot, w3dot], dtype=float)


def rhs_full(t, y, I):
    """
    ODE system for:
      y[0:3]   -> ω_body(t) = (ω1, ω2, ω3)
      y[3:6]   -> e1_space(t) (body axis 1 expressed in space coords)
      y[6:9]   -> e2_space(t)
      y[9:12]  -> e3_space(t)

    Steps each call:
    1. dω_body/dt from Euler's equations in the body frame (no torque).
    2. Construct R = [e1 e2 e3] (3x3). Columns are body axes in space coords.
    3. Compute ω_space = R @ ω_body.
    4. Evolve each body axis in space via de/dt = ω_space × e,
       which is the standard rigid-body kinematics.
    """
    omega_body = y[0:3]
    e1 = y[3:6]
    e2 = y[6:9]
    e3 = y[9:12]

    # 1. torque-free Euler
    domega_dt = euler_rhs(omega_body, I)

    # 2. rotation matrix (columns are body axes in space frame)
    R = np.column_stack([e1, e2, e3])  # (3,3)

    # 3. angular velocity in space
    omega_space = R @ omega_body  # (3,)

    # 4. body axis evolution
    def d_e_dt(e):
        return np.cross(omega_space, e)

    de1_dt = d_e_dt(e1)
    de2_dt = d_e_dt(e2)
    de3_dt = d_e_dt(e3)

    return np.hstack([domega_dt, de1_dt, de2_dt, de3_dt])


# ------------------------------------------------------------
# Geometry / visualization helpers
# ------------------------------------------------------------

def make_box(dims):
    """
    Make a rectangular box in BODY coordinates, centered at origin.

    dims = (ax, by, cz): full lengths along principal axes 1,2,3.

    Returns:
    verts_body : (8,3) array of vertex coords in body frame
    faces : list of faces, each face = list of vertex indices
    """
    ax, by, cz = dims
    xh = ax / 2.0
    yh = by / 2.0
    zh = cz / 2.0

    verts_body = np.array([
        [-xh, -yh, -zh],
        [ xh, -yh, -zh],
        [ xh,  yh, -zh],
        [-xh,  yh, -zh],
        [-xh, -yh,  zh],
        [ xh, -yh,  zh],
        [ xh,  yh,  zh],
        [-xh,  yh,  zh],
    ], dtype=float)

    faces = [
        [0,1,2,3],  # bottom (-z)
        [4,5,6,7],  # top (+z)
        [0,1,5,4],  # side
        [2,3,7,6],  # side
        [1,2,6,5],  # side
        [0,3,7,4],  # side
    ]

    return verts_body, faces


def body_to_space(R, pts_body):
    """
    Convert body-frame coords pts_body (N,3) to space coords via:
      x_space = R * x_body
    where R's columns are the body principal axes expressed in space coordinates.
    """
    return pts_body @ R.T  # (N,3)


def draw_box(ax, verts_world, faces,
             face_color=(0.6,0.6,0.9,0.5), edge_color='k'):
    """
    Draw the box faces as a Poly3DCollection with crude hidden-surface sorting.
    Returns the Poly3DCollection.
    """
    face_polys = []
    z_avgs = []
    for f in faces:
        poly = verts_world[f]
        z_avgs.append(np.mean(poly[:,2]))
        face_polys.append(poly)

    order = np.argsort(z_avgs)
    ordered_polys = [face_polys[i] for i in order]

    coll = Poly3DCollection(
        ordered_polys,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=1,
        alpha=face_color[3] if len(face_color) == 4 else 0.6
    )
    ax.add_collection3d(coll)
    return coll


def update_box_collection(coll, verts_world, faces):
    """
    Update the Poly3DCollection verts to reflect a new body orientation.
    """
    face_polys = []
    z_avgs = []
    for f in faces:
        poly = verts_world[f]
        z_avgs.append(np.mean(poly[:,2]))
        face_polys.append(poly)

    order = np.argsort(z_avgs)
    ordered_polys = [face_polys[i] for i in order]
    coll.set_verts(ordered_polys)


# ------------------------------------------------------------
# Simulation wrapper
# ------------------------------------------------------------

def simulate(I, w0, tmax=10.0, fps=60):
    """
    Integrate the pure torque-free rigid body motion.

    Initial state:
      ω_body(0)   = w0
      e1_space(0) = (1,0,0)
      e2_space(0) = (0,1,0)
      e3_space(0) = (0,0,1)

    Returns:
      t_eval           (nframes,)
      omega_body_arr   (nframes,3)
      R_t              (nframes,3,3)
                       R_t[i] has columns (e1,e2,e3) in space coords
    """
    e1_0 = np.array([1.0, 0.0, 0.0], dtype=float)
    e2_0 = np.array([0.0, 1.0, 0.0], dtype=float)
    e3_0 = np.array([0.0, 0.0, 1.0], dtype=float)

    y0 = np.hstack([w0, e1_0, e2_0, e3_0])

    def ode(t, y):
        return rhs_full(t, y, I)

    nframes = int(tmax * fps)
    t_eval = np.linspace(0.0, tmax, nframes)

    sol = solve_ivp(
        ode,
        (0.0, tmax),
        y0,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-10
    )

    if not sol.success:
        print("WARNING: ODE solver reported failure:", sol.message)

    Y = sol.y.T  # (nframes, 12)
    omega_body_arr = Y[:, 0:3]
    e1_arr         = Y[:, 3:6]
    e2_arr         = Y[:, 6:9]
    e3_arr         = Y[:, 9:12]

    # Re-orthonormalize numerically to limit drift
    def orthonormalize(e1, e2, e3):
        e1n = e1 / norm(e1)
        e2p = e2 - np.dot(e2, e1n)*e1n
        e2n = e2p / norm(e2p)
        e3n = np.cross(e1n, e2n)
        return e1n, e2n, e3n

    e1_clean = []
    e2_clean = []
    e3_clean = []
    for k in range(len(e1_arr)):
        a, b, c = orthonormalize(e1_arr[k], e2_arr[k], e3_arr[k])
        e1_clean.append(a)
        e2_clean.append(b)
        e3_clean.append(c)

    e1_clean = np.array(e1_clean)
    e2_clean = np.array(e2_clean)
    e3_clean = np.array(e3_clean)

    # Build R_t[i] = [e1 e2 e3] as columns
    R_t = np.stack(
        [
            np.column_stack([e1_clean[k], e2_clean[k], e3_clean[k]])
            for k in range(len(e1_clean))
        ],
        axis=0
    )  # (nframes,3,3)

    return t_eval, omega_body_arr, R_t


# ------------------------------------------------------------
# Poinsot ellipsoids
# ------------------------------------------------------------

def make_ellipsoid_surface(semi_axes, resolution=30):
    """
    Generate surface mesh for an ellipsoid centered at origin.

    Parameters
    ----------
    semi_axes : array-like (3,)
        Semi-axis lengths along x, y, z
    resolution : int
        Number of grid points along each spherical coordinate

    Returns
    -------
    X, Y, Z : (resolution, resolution) arrays
        Mesh coordinates for the ellipsoid surface
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    a, b, c = semi_axes

    # Parametric sphere
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale by semi-axes to get ellipsoid
    X = a * x_sphere
    Y = b * y_sphere
    Z = c * z_sphere

    return X, Y, Z


def compute_ellipsoid_intersection(I, L_mag, E, num_points=200):
    """
    Compute the intersection curve of the momentum and energy ellipsoids.

    The intersection of:
    - Momentum ellipsoid: I₁²ω₁² + I₂²ω₂² + I₃²ω₃² = L²
    - Energy ellipsoid: I₁ω₁² + I₂ω₂² + I₃ω₃² = 2E

    This is a space curve in ω-space that the angular velocity follows.

    Parameters
    ----------
    I : array-like (3,)
        Principal moments of inertia
    L_mag : float
        Magnitude of angular momentum
    E : float
        Rotational kinetic energy
    num_points : int
        Number of points to sample along the curve

    Returns
    -------
    curves : list of (N, 3) arrays
        One or more closed curves representing the intersection.
        For asymmetric tops, there are typically 1-3 disconnected curves.
    """
    I1, I2, I3 = I
    L2 = L_mag**2
    E2 = 2 * E

    # The intersection can be computed by parametrizing one component
    # and solving for the others. We'll use a shooting method:
    # fix ω3 and solve for ω1, ω2 from the two ellipsoid equations.

    curves = []

    # For each of the three possible orientations, try to find curves
    # by sweeping one coordinate and solving for the other two

    # Method: For each value of one coordinate (e.g., ω3), solve
    # the system of two ellipsoid equations for the other two coordinates

    # Determine the range for ω3 from the energy ellipsoid
    w3_max = np.sqrt(E2 / I3)
    w3_values = np.linspace(-w3_max, w3_max, num_points)

    curve_points = []

    for w3 in w3_values:
        # Given w3, solve for w1, w2:
        # I₁²w₁² + I₂²w₂² = L² - I₃²w₃²
        # I₁w₁² + I₂w₂² = 2E - I₃w₃²

        rhs_L = L2 - I3**2 * w3**2
        rhs_E = E2 - I3 * w3**2

        if rhs_L < 0 or rhs_E < 0:
            continue

        # Now we have:
        # I₁²w₁² + I₂²w₂² = rhs_L  ... (1)
        # I₁w₁² + I₂w₂² = rhs_E    ... (2)

        # From (2): w₂² = (rhs_E - I₁w₁²) / I₂
        # Substitute into (1): I₁²w₁² + I₂² * (rhs_E - I₁w₁²)/I₂ = rhs_L
        # I₁²w₁² + I₂(rhs_E - I₁w₁²) = rhs_L
        # I₁²w₁² + I₂*rhs_E - I₂*I₁w₁² = rhs_L
        # w₁²(I₁² - I₂*I₁) = rhs_L - I₂*rhs_E
        # w₁² = (rhs_L - I₂*rhs_E) / (I₁² - I₂*I₁)
        # w₁² = (rhs_L - I₂*rhs_E) / (I₁(I₁ - I₂))

        denom = I1 * (I1 - I2)
        if abs(denom) < 1e-10:
            # Degenerate case (I1 ≈ I2), need different approach
            continue

        w1_sq = (rhs_L - I2 * rhs_E) / denom

        if w1_sq < 0:
            continue

        w1 = np.sqrt(w1_sq)

        # Now get w2 from equation (2)
        w2_sq = (rhs_E - I1 * w1_sq) / I2

        if w2_sq < 0:
            continue

        w2 = np.sqrt(w2_sq)

        # Add all sign combinations (the intersection curve may have multiple branches)
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                point = np.array([s1 * w1, s2 * w2, w3])
                curve_points.append(point)

    if len(curve_points) > 0:
        curves.append(np.array(curve_points))

    return curves


# ------------------------------------------------------------
# Animation
# ------------------------------------------------------------

def animate_rigid_body(I, dims,
                       omega_body_arr, R_t,
                       fps=60,
                       outfile=None,
                       vecscale=0.2,
                       traillen=100,
                       dotpos=(0.0, 1.0, 0.0),
                       show_poinsot=False,
                       w0_initial=None,
                       tmax=10.0,
                       enable_sliders=False):
    """
    Build a figure with two or three synchronized subplots:

    Top: "Inertial Frame"
      - Body tumbles in space.
      - ω_space(t), L_space(t) arrows (red, green) with arrowheads.
      - Recent ω_space trail (red line).
      - Red marker dot (fixed point on the body) advected through space.

    Middle: "Body-Centered Frame"
      - Body frozen in body coordinates.
      - ω_body(t), L_body(t) arrows.
      - Recent ω_body trail.
      - Same red dot shown on the body surface.

    Bottom: "Poinsot Ellipsoids" (optional, shown if show_poinsot=True)
      - Shows ω in BODY FRAME where inertia tensor is diagonal
      - Two ellipsoids in body-frame ω-space:
        * Momentum ellipsoid: I₁²ω₁² + I₂²ω₂² + I₃²ω₃² = L² (green)
        * Energy ellipsoid: I₁ω₁² + I₂ω₂² + I₃ω₃² = 2E (orange, transparent)
      - Their intersection curve (yellow)
      - Trail of ω_body(t) showing the path along the intersection (red)
      - The ellipsoids are FIXED in the body frame

    Parameters
    ----------
    I : array-like (I1,I2,I3)
        Principal moments.
    dims : array-like (ax,by,cz)
        Box dimensions along the principal axes.
    omega_body_arr : (nframes,3)
        Angular velocity in body frame vs time.
    R_t : (nframes,3,3)
        Orientation matrices. Columns are body axes in space coords.
    fps : int
    outfile : str or None
        If given, an MP4 filename to save (requires ffmpeg).
    vecscale : float
        Multiplicative factor for arrow lengths.
    traillen : int
        Number of past frames to draw in the ω trails.
    dotpos : (3,)
        Fractions of HALF-length along each body axis telling us where to
        paint the red marker. Default (0,1,0) = center of +y face.
        Example: (1,0,0) = center of +x face, (1,1,1) = +corner.
    show_poinsot : bool
        If True, show the Poinsot ellipsoids in a third subplot (default: False).
    w0_initial : array-like (3,) or None
        Initial angular velocity for slider initialization. Required if enable_sliders=True.
    tmax : float
        Total simulation time (default: 10.0). Required if enable_sliders=True.
    enable_sliders : bool
        If True, add interactive sliders to adjust initial angular velocity (default: False).
    """
    I = np.array(I, dtype=float)
    dims = np.array(dims, dtype=float)
    dotpos = np.array(dotpos, dtype=float)

    nframes = R_t.shape[0]

    # Store simulation data in a mutable container for slider reset capability
    sim_data = {
        'omega_body_arr': omega_body_arr.copy(),
        'R_t': R_t.copy(),
        'nframes': nframes
    }

    # Geometry in BODY coords
    verts_body, faces = make_box(dims)

    # Marker in BODY coords:
    # dotpos are fractions of half-length, so:
    # (0,1,0) -> (0, +by/2, 0), i.e. middle of +y face.
    dot_body = 0.5 * dims * dotpos

    # Helper function to compute derived arrays from simulation data
    def compute_derived_data(omega_body_arr, R_t):
        """Compute L_body, omega_space, L_space from simulation outputs."""
        L_body_arr = omega_body_arr * I[np.newaxis, :]  # (nframes,3)
        omega_space_arr = np.einsum('nij,nj->ni', R_t, omega_body_arr)  # (nframes,3)
        L_space_arr = np.einsum('nij,nj->ni', R_t, L_body_arr)  # (nframes,3)
        return L_body_arr, omega_space_arr, L_space_arr

    # Compute angular momentum in body frame: L_body = I * ω_body (componentwise)
    # Store derived arrays in sim_data as well for easy updating
    L_body_arr, omega_space_arr, L_space_arr = compute_derived_data(
        sim_data['omega_body_arr'], sim_data['R_t']
    )
    sim_data['L_body_arr'] = L_body_arr
    sim_data['omega_space_arr'] = omega_space_arr
    sim_data['L_space_arr'] = L_space_arr

    # Frame 0 for initialization
    R0 = sim_data['R_t'][0]
    verts_world0 = body_to_space(R0, verts_body)
    dot_space0   = R0 @ dot_body

    # --------------------------------------------------------
    # Conditionally compute Poinsot ellipsoids
    # --------------------------------------------------------
    # Helper function to compute Poinsot ellipsoid data
    def compute_poinsot_data(omega_body_arr):
        """Compute ellipsoid surfaces and intersection curves."""
        w0_body = omega_body_arr[0]
        L0_body = w0_body * I
        L_mag = norm(L0_body)
        E = 0.5 * np.sum(I * w0_body**2)

        momentum_semiaxes = L_mag / I
        energy_semiaxes = np.sqrt(2 * E / I)

        X_momentum, Y_momentum, Z_momentum = make_ellipsoid_surface(momentum_semiaxes, resolution=30)
        X_energy, Y_energy, Z_energy = make_ellipsoid_surface(energy_semiaxes, resolution=30)

        intersection_curves = compute_ellipsoid_intersection(I, L_mag, E, num_points=200)

        return {
            'X_momentum': X_momentum, 'Y_momentum': Y_momentum, 'Z_momentum': Z_momentum,
            'X_energy': X_energy, 'Y_energy': Y_energy, 'Z_energy': Z_energy,
            'intersection_curves': intersection_curves,
            'momentum_semiaxes': momentum_semiaxes,
            'energy_semiaxes': energy_semiaxes
        }

    if show_poinsot:
        # Using initial conditions (conserved throughout motion)
        poinsot_data = compute_poinsot_data(sim_data['omega_body_arr'])
        sim_data['poinsot'] = poinsot_data

        # Extract computed ellipsoid data
        X_momentum = poinsot_data['X_momentum']
        Y_momentum = poinsot_data['Y_momentum']
        Z_momentum = poinsot_data['Z_momentum']
        X_energy = poinsot_data['X_energy']
        Y_energy = poinsot_data['Y_energy']
        Z_energy = poinsot_data['Z_energy']
        intersection_curves = poinsot_data['intersection_curves']
        momentum_semiaxes = poinsot_data['momentum_semiaxes']
        energy_semiaxes = poinsot_data['energy_semiaxes']

    # --------------------------------------------------------
    # Figure with two or three stacked 3D subplots
    # --------------------------------------------------------
    # Adjust figure size and layout based on whether we're showing sliders
    if enable_sliders:
        if show_poinsot:
            fig = plt.figure(figsize=(7,15.5))
            # Leave space at bottom for sliders (about 1.5 units)
            gs = fig.add_gridspec(3, 1, height_ratios=[1,1,1], bottom=0.12, top=0.98)
        else:
            fig = plt.figure(figsize=(7,11.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[1,1], bottom=0.15, top=0.98)
    else:
        if show_poinsot:
            fig = plt.figure(figsize=(7,14))
            gs = fig.add_gridspec(3, 1, height_ratios=[1,1,1])
        else:
            fig = plt.figure(figsize=(7,10))
            gs = fig.add_gridspec(2, 1, height_ratios=[1,1])

    ax_inertial = fig.add_subplot(gs[0,0], projection='3d')
    ax_body     = fig.add_subplot(gs[1,0], projection='3d')
    ax_poinsot  = fig.add_subplot(gs[2,0], projection='3d') if show_poinsot else None

    def setup_ax(ax, title, lim=None):
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=30)
        if lim is None:
            max_dim = max(dims)
            lim = 1.8 * max_dim
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

    setup_ax(ax_inertial, "Inertial Frame")
    setup_ax(ax_body,     "Body-Centered Frame")

    if show_poinsot:
        # For Poinsot frame, use the larger of the two ellipsoid extents
        poinsot_lim = 1.2 * max(np.max(momentum_semiaxes), np.max(energy_semiaxes))
        setup_ax(ax_poinsot,  "Poinsot Ellipsoids (ω-space)", lim=poinsot_lim)

    # --------------------------------------------------------
    # Inertial frame artists
    # --------------------------------------------------------
    box_coll_inertial = draw_box(
        ax_inertial, verts_world0, faces,
        face_color=(0.6,0.6,0.9,0.5),
        edge_color='k'
    )

    origin = np.array([0.0,0.0,0.0])

    w_space0 = omega_space_arr[0]
    L_space0 = L_space_arr[0]

    w_space_vec0 = vecscale * w_space0
    L_space_vec0 = vecscale * L_space0

    w_quiv_space = ax_inertial.quiver(
        origin[0], origin[1], origin[2],
        w_space_vec0[0], w_space_vec0[1], w_space_vec0[2],
        color='red',
        arrow_length_ratio=0.15,
        linewidth=2
    )
    L_quiv_space = ax_inertial.quiver(
        origin[0], origin[1], origin[2],
        L_space_vec0[0], L_space_vec0[1], L_space_vec0[2],
        color='green',
        arrow_length_ratio=0.15,
        linewidth=2
    )

    w_label_space = ax_inertial.text(
        w_space_vec0[0]*1.05,
        w_space_vec0[1]*1.05,
        w_space_vec0[2]*1.05,
        r'$\omega$',
        color='red'
    )
    L_label_space = ax_inertial.text(
        L_space_vec0[0]*1.05,
        L_space_vec0[1]*1.05,
        L_space_vec0[2]*1.05,
        r'$\mathbf{L}$',
        color='green'
    )

    # ω trail in inertial frame
    trail_line_space, = ax_inertial.plot([], [], [], lw=2, color='red', alpha=0.5)
    w_space_history = []

    # Red marker dot in inertial frame
    dot_point_space, = ax_inertial.plot(
        [dot_space0[0]], [dot_space0[1]], [dot_space0[2]],
        'o', color='red', markersize=8
    )

    # --------------------------------------------------------
    # Body-centered frame artists
    # --------------------------------------------------------
    box_coll_body = draw_box(
        ax_body, verts_body, faces,
        face_color=(0.6,0.6,0.9,0.5),
        edge_color='k'
    )

    w_body0 = omega_body_arr[0]
    L_body0 = L_body_arr[0]

    w_body_vec0 = vecscale * w_body0
    L_body_vec0 = vecscale * L_body0

    w_quiv_body = ax_body.quiver(
        origin[0], origin[1], origin[2],
        w_body_vec0[0], w_body_vec0[1], w_body_vec0[2],
        color='red',
        arrow_length_ratio=0.15,
        linewidth=2
    )
    L_quiv_body = ax_body.quiver(
        origin[0], origin[1], origin[2],
        L_body_vec0[0], L_body_vec0[1], L_body_vec0[2],
        color='green',
        arrow_length_ratio=0.15,
        linewidth=2
    )

    w_label_body = ax_body.text(
        w_body_vec0[0]*1.05,
        w_body_vec0[1]*1.05,
        w_body_vec0[2]*1.05,
        r'$\omega$',
        color='red'
    )
    L_label_body = ax_body.text(
        L_body_vec0[0]*1.05,
        L_body_vec0[1]*1.05,
        L_body_vec0[2]*1.05,
        r'$\mathbf{L}$',
        color='green'
    )

    # ω trail in body frame
    trail_line_body, = ax_body.plot([], [], [], lw=2, color='red', alpha=0.5)
    w_body_history = []

    # Red marker dot in body coords (fixed point on the box)
    dot_point_body, = ax_body.plot(
        [dot_body[0]], [dot_body[1]], [dot_body[2]],
        'o', color='red', markersize=8
    )

    # --------------------------------------------------------
    # Poinsot frame artists (optional)
    # --------------------------------------------------------
    if show_poinsot:
        # Draw the two ellipsoids
        momentum_surf = ax_poinsot.plot_surface(
            X_momentum, Y_momentum, Z_momentum,
            color='green', alpha=0.3, edgecolor='none'
        )

        energy_surf = ax_poinsot.plot_surface(
            X_energy, Y_energy, Z_energy,
            color='orange', alpha=0.25, edgecolor='none'
        )

        # Draw the intersection curve(s)
        intersection_artists = []
        for curve in intersection_curves:
            if len(curve) > 0:
                line, = ax_poinsot.plot(
                    curve[:, 0], curve[:, 1], curve[:, 2],
                    color='yellow', linewidth=2, alpha=0.7
                )
                intersection_artists.append(line)

        # ω trail in Poinsot frame (ω_body trajectory)
        trail_line_poinsot, = ax_poinsot.plot([], [], [], lw=2, color='red', alpha=0.7)
        w_poinsot_history = []

        # Current ω point
        w_point_poinsot, = ax_poinsot.plot(
            [w0_body[0]], [w0_body[1]], [w0_body[2]],
            'o', color='red', markersize=8
        )

        # Add labels to Poinsot frame
        ax_poinsot.text2D(0.05, 0.95, r'Green: $L^2$ ellipsoid', transform=ax_poinsot.transAxes,
                          color='green', fontsize=10)
        ax_poinsot.text2D(0.05, 0.90, r'Orange: $2E$ ellipsoid', transform=ax_poinsot.transAxes,
                          color='orange', fontsize=10)
        ax_poinsot.text2D(0.05, 0.85, r'Red: $\omega(t)$ trajectory', transform=ax_poinsot.transAxes,
                          color='red', fontsize=10)
    else:
        # Placeholder values when Poinsot is not shown
        momentum_surf = None
        energy_surf = None
        intersection_artists = []
        trail_line_poinsot = None
        w_poinsot_history = []
        w_point_poinsot = None

    # --------------------------------------------------------
    # Interactive Sliders and Reset Button (optional)
    # --------------------------------------------------------
    if enable_sliders:
        if w0_initial is None:
            raise ValueError("w0_initial must be provided when enable_sliders=True")

        # Create slider axes at the bottom of the figure
        slider_left = 0.15
        slider_width = 0.7
        slider_height = 0.02
        slider_spacing = 0.03

        ax_slider_w1 = fig.add_axes([slider_left, 0.08, slider_width, slider_height])
        ax_slider_w2 = fig.add_axes([slider_left, 0.05, slider_width, slider_height])
        ax_slider_w3 = fig.add_axes([slider_left, 0.02, slider_width, slider_height])

        # Create sliders with initial values from w0_initial
        slider_w1 = Slider(ax_slider_w1, r'$\omega_1$', -20.0, 20.0,
                          valinit=w0_initial[0], valstep=0.1)
        slider_w2 = Slider(ax_slider_w2, r'$\omega_2$', -20.0, 20.0,
                          valinit=w0_initial[1], valstep=0.1)
        slider_w3 = Slider(ax_slider_w3, r'$\omega_3$', -20.0, 20.0,
                          valinit=w0_initial[2], valstep=0.1)

        # Add vertical lines at zero for each slider to make zero more visible
        for ax_slider in [ax_slider_w1, ax_slider_w2, ax_slider_w3]:
            ax_slider.axvline(0, color='black', linewidth=2.0, linestyle='-', alpha=0.6, zorder=10)

        # Create reset button
        ax_button = fig.add_axes([0.45, 0.11, 0.1, 0.02])
        button_reset = Button(ax_button, 'Reset', color='lightblue', hovercolor='skyblue')

        # Store reference to the animation object so we can restart it
        anim_ref = {'anim': None}

        def reset_simulation(event):
            """Callback for reset button: recompute simulation with new w0."""
            # Get new initial conditions from sliders (in BODY FRAME)
            # These are the angular velocity components along the principal axes
            new_w0 = np.array([slider_w1.val, slider_w2.val, slider_w3.val], dtype=float)

            # Recompute simulation
            _, new_omega_body_arr, new_R_t = simulate(I, new_w0, tmax=tmax, fps=fps)

            # Update simulation data in place
            sim_data['omega_body_arr'] = new_omega_body_arr
            sim_data['R_t'] = new_R_t
            sim_data['nframes'] = new_R_t.shape[0]

            # Recompute derived arrays
            new_L_body_arr, new_omega_space_arr, new_L_space_arr = compute_derived_data(
                new_omega_body_arr, new_R_t
            )
            sim_data['L_body_arr'] = new_L_body_arr
            sim_data['omega_space_arr'] = new_omega_space_arr
            sim_data['L_space_arr'] = new_L_space_arr

            # Recompute Poinsot ellipsoids if needed
            if show_poinsot:
                new_poinsot_data = compute_poinsot_data(new_omega_body_arr)
                sim_data['poinsot'] = new_poinsot_data

                # Update Poinsot ellipsoid surfaces
                momentum_surf._vec = np.array([
                    new_poinsot_data['X_momentum'],
                    new_poinsot_data['Y_momentum'],
                    new_poinsot_data['Z_momentum']
                ])
                energy_surf._vec = np.array([
                    new_poinsot_data['X_energy'],
                    new_poinsot_data['Y_energy'],
                    new_poinsot_data['Z_energy']
                ])

                # Update intersection curves
                for artist in intersection_artists:
                    artist.remove()
                intersection_artists.clear()
                for curve in new_poinsot_data['intersection_curves']:
                    if len(curve) > 0:
                        line, = ax_poinsot.plot(
                            curve[:, 0], curve[:, 1], curve[:, 2],
                            color='yellow', linewidth=2, alpha=0.7
                        )
                        intersection_artists.append(line)

            # Clear history for trails
            w_space_history.clear()
            w_body_history.clear()
            if show_poinsot:
                w_poinsot_history.clear()

            # Restart animation from frame 0
            if anim_ref['anim'] is not None:
                anim_ref['anim'].event_source.stop()

                # Reset to initial state by calling init function
                init()

                # Reset the frame generator to start from 0
                # _iter_gen must be a callable that returns an iterator
                anim_ref['anim']._iter_gen = lambda: iter(range(sim_data['nframes']))

                # Restart the animation
                anim_ref['anim'].event_source.start()

            fig.canvas.draw_idle()

        button_reset.on_clicked(reset_simulation)
    else:
        slider_w1 = None
        slider_w2 = None
        slider_w3 = None
        button_reset = None

    # --------------------------------------------------------
    # Animation functions
    # --------------------------------------------------------

    def init():
        artists = [box_coll_inertial, w_quiv_space, L_quiv_space,
                   w_label_space, L_label_space, trail_line_space, dot_point_space,
                   box_coll_body, w_quiv_body, L_quiv_body,
                   w_label_body, L_label_body, trail_line_body, dot_point_body]
        if show_poinsot:
            artists.extend([momentum_surf, energy_surf, trail_line_poinsot, w_point_poinsot])
            artists.extend(intersection_artists)
        return tuple(artists)

    def update(frame):
        nonlocal w_space_history, w_body_history, w_poinsot_history

        # Read from sim_data to support dynamic updates from sliders
        R = sim_data['R_t'][frame]                 # (3,3)
        w_body = sim_data['omega_body_arr'][frame] # (3,)
        L_body = w_body * I                        # componentwise
        w_space = sim_data['omega_space_arr'][frame]
        L_space = sim_data['L_space_arr'][frame]

        # ----- Inertial frame update -----
        verts_world = body_to_space(R, verts_body)
        update_box_collection(box_coll_inertial, verts_world, faces)

        w_space_vec = vecscale * w_space
        L_space_vec = vecscale * L_space

        w_quiv_space.set_segments(
            [np.vstack([origin, origin + w_space_vec])]
        )
        w_quiv_space._segments3d = [np.vstack([origin, origin + w_space_vec])]

        L_quiv_space.set_segments(
            [np.vstack([origin, origin + L_space_vec])]
        )
        L_quiv_space._segments3d = [np.vstack([origin, origin + L_space_vec])]

        w_label_space.set_position((w_space_vec[0]*1.05, w_space_vec[1]*1.05))
        w_label_space.set_3d_properties(w_space_vec[2]*1.05)
        L_label_space.set_position((L_space_vec[0]*1.05, L_space_vec[1]*1.05))
        L_label_space.set_3d_properties(L_space_vec[2]*1.05)

        # update ω trail in inertial frame
        w_space_history.append(w_space_vec.copy())
        if len(w_space_history) > traillen:
            w_space_history = w_space_history[-traillen:]
        w_hist_arr = np.array(w_space_history)
        trail_line_space.set_data(w_hist_arr[:,0], w_hist_arr[:,1])
        trail_line_space.set_3d_properties(w_hist_arr[:,2])

        # move inertial red dot
        dot_space = R @ dot_body
        dot_point_space.set_data([dot_space[0]], [dot_space[1]])
        dot_point_space.set_3d_properties([dot_space[2]])

        # ----- Body-centered frame update -----
        w_body_vec = vecscale * w_body
        L_body_vec = vecscale * L_body

        w_quiv_body.set_segments(
            [np.vstack([origin, origin + w_body_vec])]
        )
        w_quiv_body._segments3d = [np.vstack([origin, origin + w_body_vec])]

        L_quiv_body.set_segments(
            [np.vstack([origin, origin + L_body_vec])]
        )
        L_quiv_body._segments3d = [np.vstack([origin, origin + L_body_vec])]

        w_label_body.set_position((w_body_vec[0]*1.05, w_body_vec[1]*1.05))
        w_label_body.set_3d_properties(w_body_vec[2]*1.05)
        L_label_body.set_position((L_body_vec[0]*1.05, L_body_vec[1]*1.05))
        L_label_body.set_3d_properties(L_body_vec[2]*1.05)

        # update ω trail in body frame
        w_body_history.append(w_body_vec.copy())
        if len(w_body_history) > traillen:
            w_body_history = w_body_history[-traillen:]
        w_hist_body_arr = np.array(w_body_history)
        trail_line_body.set_data(w_hist_body_arr[:,0], w_hist_body_arr[:,1])
        trail_line_body.set_3d_properties(w_hist_body_arr[:,2])

        # dot_point_body doesn't move in body coords

        # ----- Poinsot frame update -----
        if show_poinsot:
            # Update the ω trajectory in ω-space
            w_poinsot_history.append(w_body.copy())
            if len(w_poinsot_history) > traillen:
                w_poinsot_history = w_poinsot_history[-traillen:]
            w_hist_poinsot_arr = np.array(w_poinsot_history)
            trail_line_poinsot.set_data(w_hist_poinsot_arr[:,0], w_hist_poinsot_arr[:,1])
            trail_line_poinsot.set_3d_properties(w_hist_poinsot_arr[:,2])

            # Update current ω point
            w_point_poinsot.set_data([w_body[0]], [w_body[1]])
            w_point_poinsot.set_3d_properties([w_body[2]])

        artists = [box_coll_inertial, w_quiv_space, L_quiv_space,
                   w_label_space, L_label_space, trail_line_space, dot_point_space,
                   box_coll_body, w_quiv_body, L_quiv_body,
                   w_label_body, L_label_body, trail_line_body, dot_point_body]
        if show_poinsot:
            artists.extend([momentum_surf, energy_surf, trail_line_poinsot, w_point_poinsot])
            artists.extend(intersection_artists)
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=nframes,
        interval=1000.0/fps,
        blit=False
    )

    # Store animation reference for slider reset functionality
    if enable_sliders:
        anim_ref['anim'] = anim

    if outfile is not None:
        try:
            anim.save(outfile, fps=fps, dpi=150)
            print(f"Saved animation to {outfile}")
        except Exception as e:
            print(f"Could not save animation: {e}")

    plt.show()


# ------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------

def build_initial_w(args, I):
    """
    Decide the initial angular velocity in the BODY frame.

    Priority:
    - If user passed --w0, use that directly.
    - Else:
        --unstable : mostly about axis 2 (intermediate inertia, unstable)
        --stable3  : mostly about axis 3 (largest inertia, stable)
        --stable1  : mostly about axis 1 (smallest inertia, also stable; default)

    We also add a small perturbation in the off-axes so the motion isn't
    perfectly aligned and boring.
    """
    if args.w0 is not None:
        return np.array(args.w0, dtype=float)

    mag = args.mag
    eps = 0.05 * mag

    if args.unstable:
        # mostly about axis 2 (the "book toss" unstable axis)
        return np.array([eps, mag, eps], dtype=float)
    if args.stable3:
        # mostly about axis 3, typically largest I, very stable
        return np.array([eps, eps, mag], dtype=float)

    # default or --stable1:
    # mostly about axis 1, typically smallest I, also stable
    return np.array([mag, eps, eps], dtype=float)


def calculate_dims_from_I(I):
    """
    Calculate box dimensions [a, b, c] that produce the given principal moments
    of inertia for a uniform-density box.

    For a uniform box: I1 ∝ (b² + c²), I2 ∝ (a² + c²), I3 ∝ (a² + b²)

    Returns None if the moments don't satisfy the triangle inequality
    (required for any physical rigid body).
    """
    I1, I2, I3 = I

    # Check triangle inequality
    if I1 + I2 <= I3 or I1 + I3 <= I2 or I2 + I3 <= I1:
        return None

    # Solve the system:
    # b² + c² = I1
    # a² + c² = I2
    # a² + b² = I3
    #
    # Adding all three: 2(a² + b² + c²) = I1 + I2 + I3
    # So: a² + b² + c² = (I1 + I2 + I3) / 2
    sum_all = (I1 + I2 + I3) / 2.0

    # Then: a² = sum_all - I1, b² = sum_all - I2, c² = sum_all - I3
    a_sq = sum_all - I1
    b_sq = sum_all - I2
    c_sq = sum_all - I3

    # All must be positive
    if a_sq <= 0 or b_sq <= 0 or c_sq <= 0:
        return None

    return [np.sqrt(a_sq), np.sqrt(b_sq), np.sqrt(c_sq)]


def parse_args():
    p = argparse.ArgumentParser(
        description="Animate torque-free rigid body rotation (no quaternions)."
    )

    p.add_argument(
        "--dims", nargs=3, type=float, default=None,
        metavar=('AX','BY','CZ'),
        help="Box dimensions along body principal axes. If not specified, "
             "automatically calculated from --I to match a uniform-density box."
    )

    p.add_argument(
        "--I", nargs=3, type=float, default=[1.0, 4.0, 4.9],
        metavar=('I1','I2','I3'),
        help="Principal moments of inertia (default 1 4 4.9)."
    )

    p.add_argument(
        "--w0", nargs=3, type=float, default=None,
        metavar=('W1','W2','W3'),
        help="Initial angular velocity in BODY frame. Overrides presets."
    )

    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--stable1", action="store_true",
        help="Spin mostly about axis 1 (often smallest inertia, stable)."
    )
    g.add_argument(
        "--unstable", action="store_true",
        help="Spin mostly about axis 2 (intermediate inertia, unstable)."
    )
    g.add_argument(
        "--stable3", action="store_true",
        help="Spin mostly about axis 3 (largest inertia, stable)."
    )

    p.add_argument(
        "--mag", type=float, default=10.0,
        help="Base spin magnitude for presets (default 10.0)."
    )

    p.add_argument(
        "--tmax", type=float, default=10.0,
        help="Total simulated time (default 10.0)."
    )

    p.add_argument(
        "--fps", type=int, default=60,
        help="Frames per second (default 60)."
    )

    p.add_argument(
        "--outfile", type=str, default=None,
        help="Optional mp4 filename. Requires ffmpeg."
    )

    p.add_argument(
        "--vecscale", type=float, default=0.2,
        help="Scale factor for ω and L arrows (default 0.2)."
    )

    p.add_argument(
        "--traillen", type=int, default=100,
        help="Number of recent frames for the ω trails (default 100)."
    )

    p.add_argument(
        "--dotpos", nargs=3, type=float, default=[0.0, 1.0, 0.0],
        metavar=('X','Y','Z'),
        help=(
            "Location of the red marker in BODY coordinates, "
            "given as fractions of HALF the box size. "
            "Default 0 1 0 = center of +y face. "
            "1 0 0 = +x face center, 0 0 1 = +z face center, 1 1 1 = +corner."
        )
    )

    p.add_argument(
        "--poinsot", action="store_true",
        help="Show Poinsot ellipsoids in a third subplot (default: False)."
    )

    p.add_argument(
        "--sliders", action="store_true",
        help="Add interactive sliders to dynamically adjust initial angular velocity (default: False)."
    )

    return p.parse_args()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    args = parse_args()

    I = np.array(args.I, dtype=float)

    # If dims not specified, calculate from I to match uniform-density box
    if args.dims is None:
        dims = calculate_dims_from_I(I)
        if dims is None:
            print("Warning: Moments of inertia violate triangle inequality.")
            print("Using default dims=[2.0, 1.0, 0.5] instead.")
            dims = [2.0, 1.0, 0.5]
        dims = np.array(dims, dtype=float)
    else:
        dims = np.array(args.dims, dtype=float)

    # choose initial ω in the body frame
    w0 = build_initial_w(args, I)

    # integrate torque-free rigid body dynamics
    t_arr, omega_body_arr, R_t = simulate(
        I, w0,
        tmax=args.tmax,
        fps=args.fps
    )

    # animate
    animate_rigid_body(
        I, dims,
        omega_body_arr, R_t,
        fps=args.fps,
        outfile=args.outfile,
        vecscale=args.vecscale,
        traillen=args.traillen,
        dotpos=args.dotpos,
        show_poinsot=args.poinsot,
        w0_initial=w0,
        tmax=args.tmax,
        enable_sliders=args.sliders
    )


if __name__ == "__main__":
    main()

