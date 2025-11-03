#!/usr/bin/env python3
"""
Animate torque-free rigid body rotation with unequal principal moments of
inertia.

-rtf, 110325, written with the assistance of Claude Code.

We produce TWO synchronized 3D views:

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

Dependencies: numpy, scipy, matplotlib
"""

import argparse
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib import animation
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
# Animation
# ------------------------------------------------------------

def animate_rigid_body(I, dims,
                       omega_body_arr, R_t,
                       fps=60,
                       outfile=None,
                       vecscale=0.2,
                       traillen=100,
                       dotpos=(0.0, 1.0, 0.0)):
    """
    Build a figure with two synchronized subplots:

    Top: "Inertial Frame"
      - Body tumbles in space.
      - ω_space(t), L_space(t) arrows (red, green) with arrowheads.
      - Recent ω_space trail (red line).
      - Red marker dot (fixed point on the body) advected through space.

    Bottom: "Body-Centered Frame"
      - Body frozen in body coordinates.
      - ω_body(t), L_body(t) arrows.
      - Recent ω_body trail.
      - Same red dot shown on the body surface.

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
    """
    I = np.array(I, dtype=float)
    dims = np.array(dims, dtype=float)
    dotpos = np.array(dotpos, dtype=float)

    nframes = R_t.shape[0]

    # Geometry in BODY coords
    verts_body, faces = make_box(dims)

    # Marker in BODY coords:
    # dotpos are fractions of half-length, so:
    # (0,1,0) -> (0, +by/2, 0), i.e. middle of +y face.
    dot_body = 0.5 * dims * dotpos

    # Compute angular momentum in body frame: L_body = I * ω_body (componentwise)
    L_body_arr = omega_body_arr * I[np.newaxis, :]  # (nframes,3)

    # Convert ω_body and L_body to space frame for inertial view
    omega_space_arr = np.einsum('nij,nj->ni', R_t, omega_body_arr)  # (nframes,3)
    L_space_arr     = np.einsum('nij,nj->ni', R_t, L_body_arr)      # (nframes,3)

    # Frame 0 for initialization
    R0 = R_t[0]
    verts_world0 = body_to_space(R0, verts_body)
    dot_space0   = R0 @ dot_body

    # --------------------------------------------------------
    # Figure with two stacked 3D subplots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(7,10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1,1])

    ax_inertial = fig.add_subplot(gs[0,0], projection='3d')
    ax_body     = fig.add_subplot(gs[1,0], projection='3d')

    def setup_ax(ax, title):
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=20, azim=30)
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
    # Animation functions
    # --------------------------------------------------------

    def init():
        return (box_coll_inertial, w_quiv_space, L_quiv_space,
                w_label_space, L_label_space, trail_line_space, dot_point_space,
                box_coll_body, w_quiv_body, L_quiv_body,
                w_label_body, L_label_body, trail_line_body, dot_point_body)

    def update(frame):
        nonlocal w_space_history, w_body_history

        R = R_t[frame]                 # (3,3)
        w_body = omega_body_arr[frame] # (3,)
        L_body = w_body * I            # componentwise
        w_space = omega_space_arr[frame]
        L_space = L_space_arr[frame]

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

        return (box_coll_inertial, w_quiv_space, L_quiv_space,
                w_label_space, L_label_space, trail_line_space, dot_point_space,
                box_coll_body, w_quiv_body, L_quiv_body,
                w_label_body, L_label_body, trail_line_body, dot_point_body)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=nframes,
        interval=1000.0/fps,
        blit=False
    )

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Animate torque-free rigid body rotation (no quaternions)."
    )

    p.add_argument(
        "--dims", nargs=3, type=float, default=[1.4, 1.0, 0.3],
        metavar=('AX','BY','CZ'),
        help="Box dimensions along body principal axes (default 1.4 1.0 0.3). "
             "Defaults approximately match I=[1,2,3] for a uniform-density box."
    )

    p.add_argument(
        "--I", nargs=3, type=float, default=[1.0, 2.0, 3.0],
        metavar=('I1','I2','I3'),
        help="Principal moments of inertia (default 1 2 3)."
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

    return p.parse_args()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    args = parse_args()

    I    = np.array(args.I,    dtype=float)
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
        dotpos=args.dotpos
    )


if __name__ == "__main__":
    main()

