# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based rigid body rotation simulator that visualizes torque-free rotation of objects with unequal principal moments of inertia. The code demonstrates classical physics phenomena like the tennis-racket instability without using quaternions.

**Key Feature:** The simulator produces **two synchronized 3D views** side-by-side:
1. **Inertial Frame** - Watch the body tumble in space with fixed angular momentum
2. **Body-Centered Frame** - Ride along with the body and see how ω and L evolve in body coordinates

## Running the Simulation

```bash
# Rotation about intermediate unstable axis for 15 time units
python rigid_rotor.py --unstable --tmax 15

# Save animation to file (requires ffmpeg)
python rigid_rotor.py --unstable --outfile tumble.mp4

# Customize the red marker dot position (fractions of half-box dimensions)
python rigid_rotor.py --unstable --dotpos 1 1 1  # corner of box
```

**Dependencies:**
```bash
pip install numpy scipy matplotlib

# For MP4 output, ffmpeg is required:
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

## Code Architecture

### Physical Model

The simulation integrates a 12-dimensional state vector consisting of:
- **ω_body(t)**: 3 components of angular velocity in the body frame
- **e₁(t), e₂(t), e₃(t)**: 9 components representing the 3 body principal axes expressed in space coordinates

This approach avoids quaternions by directly integrating the body axes as vectors using:
- **Euler's equations** for dω/dt in the body frame (euler_rhs)
- **Axis evolution** via de/dt = ω_space × e for each principal axis (rhs_full)

The rotation matrix R(t) = [e₁ e₂ e₃] transforms between body and space frames at each timestep.

### Key Functions

**Physics Integration:**
- `euler_rhs(omega_body, I)`: Implements torque-free Euler's equations in body frame
- `rhs_full(t, y, I)`: Full ODE system combining Euler's equations with axis evolution
- `simulate(I, w0, tmax, fps)`: Uses scipy.integrate.solve_ivp to integrate the system, with periodic re-orthonormalization to prevent numerical drift

**Visualization:**

- `animate_rigid_body()`: Creates dual-view matplotlib animation with synchronized subplots:

  **Top subplot (Inertial Frame):**
  - Rectangular box tumbling in space (z-sorted for hidden surface removal)
  - Red arrow with arrowhead showing ω_space (angular velocity in space frame)
  - Green arrow with arrowhead showing L_space (angular momentum, conserved and fixed)
  - Red trail showing recent path of ω_space (configurable via --traillen)
  - Red marker dot painted on the body surface, advecting through space

  **Bottom subplot (Body-Centered Frame):**
  - Rectangular box frozen in body coordinates
  - Red arrow showing ω_body (angular velocity in body frame)
  - Green arrow showing L_body (angular momentum in body frame)
  - Red trail showing recent path of ω_body
  - Red marker dot at fixed position on body surface (specified by --dotpos)

**Geometry:**
- `make_box(dims)`: Generates box vertices and faces in body frame
- `body_to_space(R, pts_body)`: Transforms body coordinates to space using rotation matrix
- `draw_box()` and `update_box_collection()`: Render box with crude z-sorting for hidden surfaces

### Important CLI Parameters

**Physical Parameters:**
- `--I I1 I2 I3`: Principal moments of inertia (default: 1.0 2.0 3.0)
- `--dims AX BY CZ`: Box dimensions along principal axes (default: 2.0 1.0 0.5)
- `--w0 W1 W2 W3`: Initial angular velocity in body frame (overrides presets)

**Initial Condition Presets:**
- `--stable1`: Spin mostly about axis 1 (smallest inertia, stable) - DEFAULT
- `--unstable`: Spin mostly about axis 2 (intermediate inertia, unstable - tennis racket effect)
- `--stable3`: Spin mostly about axis 3 (largest inertia, stable)
- `--mag`: Base spin magnitude for presets (default: 10.0)

**Visualization Controls:**
- `--vecscale`: Arrow length scaling factor (default: 0.2)
- `--traillen`: Number of frames in ω trails (default: 100)
- `--dotpos X Y Z`: Red marker position in body frame as fractions of half-box dimensions (default: 0 1 0)
  - `0 1 0` = center of +y face (default)
  - `1 0 0` = center of +x face
  - `0 0 1` = center of +z face
  - `1 1 1` = corner of box

**Simulation Controls:**
- `--tmax`: Total simulation time (default: 10.0)
- `--fps`: Frames per second (default: 60)
- `--outfile`: Save to MP4 file (requires ffmpeg)

## Physics Interpretation

The dual-frame visualization reveals complementary aspects of rigid body dynamics:

**Inertial Frame (top):**
- The green L vector remains fixed in space (conservation of angular momentum)
- The red ω vector precesses around L, tracing out complex patterns
- The separation between ω and L reveals non-trivial dynamics
- The red marker dot shows how material points on the body trace paths through space

**Body-Centered Frame (bottom):**
- Both ω and L vectors evolve in body coordinates
- For stable rotation (axes 1 or 3): ω stays nearly aligned with one principal axis
- For unstable rotation (axis 2): ω undergoes dramatic tumbling, displaying the tennis-racket effect
- The red marker dot stays fixed, helping you confirm you're riding with the body

**Key Observations:**
- Rotation about smallest (I₁) or largest (I₃) inertia axes is stable
- Rotation about the intermediate axis (I₂) is unstable, causing dramatic tumbling
- The red trails in both frames make precession patterns visible
- Energy and |L| are conserved (verified numerically with re-orthonormalization)
