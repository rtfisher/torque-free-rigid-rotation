# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based rigid body rotation simulator that visualizes torque-free rotation of objects with unequal principal moments of inertia. The code demonstrates classical physics phenomena like the tennis-racket instability without using quaternions.

**Key Feature:** The simulator produces **two synchronized 3D views** by default, or **three** with the `--poinsot` flag:
1. **Inertial Frame** - Watch the body tumble in space with fixed angular momentum
2. **Body-Centered Frame** - Ride along with the body and see how ω and L evolve in body coordinates
3. **Poinsot Ellipsoids** (optional) - See the geometric interpretation in ω-space with the momentum and energy ellipsoids and their intersection curve

## Running the Simulation

```bash
# Rotation about intermediate unstable axis for 15 time units
python rigid_rotor.py --unstable --tmax 15

# Add Poinsot ellipsoids visualization (3rd frame)
python rigid_rotor.py --unstable --tmax 15 --poinsot

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

- `animate_rigid_body(show_poinsot=False)`: Creates two or three-view matplotlib animation with synchronized subplots:

  **Top subplot (Inertial Frame):**
  - Rectangular box tumbling in space (z-sorted for hidden surface removal)
  - Red arrow with arrowhead showing ω_space (angular velocity in space frame)
  - Green arrow with arrowhead showing L_space (angular momentum, conserved and fixed)
  - Red trail showing recent path of ω_space (configurable via --traillen)
  - Red marker dot painted on the body surface, advecting through space

  **Middle subplot (Body-Centered Frame):**
  - Rectangular box frozen in body coordinates
  - Red arrow showing ω_body (angular velocity in body frame)
  - Green arrow showing L_body (angular momentum in body frame)
  - Red trail showing recent path of ω_body
  - Red marker dot at fixed position on body surface (specified by --dotpos)

  **Bottom subplot (Poinsot Ellipsoids - optional, shown with --poinsot flag):**
  - Shows ω in **body frame** where inertia tensor is diagonal
  - Green ellipsoid representing momentum conservation: I₁²ω₁² + I₂²ω₂² + I₃²ω₃² = L²
  - Orange (transparent) ellipsoid representing energy conservation: I₁ω₁² + I₂ω₂² + I₃ω₃² = 2E
  - Yellow curve showing the intersection of the two ellipsoids
  - Red trail showing the trajectory of ω_body(t) along the intersection curve
  - Demonstrates Poinsot's geometric interpretation of rigid body rotation
  - Ellipsoids are fixed in this body-frame representation

**Poinsot Construction:**
- `make_ellipsoid_surface(semi_axes)`: Generates surface mesh for ellipsoids in ω-space
- `compute_ellipsoid_intersection(I, L_mag, E)`: Computes the intersection curve of momentum and energy ellipsoids
  - Solves the system of two quadratic constraints in ω-space
  - Returns space curves along which ω_body(t) must move

**Geometry:**
- `make_box(dims)`: Generates box vertices and faces in body frame
- `body_to_space(R, pts_body)`: Transforms body coordinates to space using rotation matrix
- `draw_box()` and `update_box_collection()`: Render box with crude z-sorting for hidden surfaces

### Important CLI Parameters

**Physical Parameters:**
- `--I I1 I2 I3`: Principal moments of inertia (default: 1.0 4.0 4.9)
- `--dims AX BY CZ`: Box dimensions along principal axes (auto-calculated from I if not specified to match uniform-density box)
- `--w0 W1 W2 W3`: Initial angular velocity in body frame (overrides presets)

**Initial Condition Presets:**
- `--stable1`: Spin mostly about axis 1 (smallest inertia, stable) - DEFAULT
- `--unstable`: Spin mostly about axis 2 (intermediate inertia, unstable - tennis racket effect)
- `--stable3`: Spin mostly about axis 3 (largest inertia, stable)
- `--mag`: Base spin magnitude for presets (default: 10.0)

**Visualization Controls:**
- `--poinsot`: Show Poinsot ellipsoids in a third subplot (default: False)
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

The visualization reveals complementary aspects of rigid body dynamics. By default, two frames are shown; add `--poinsot` for the third frame:

**Inertial Frame (top):**
- The green L vector remains fixed in space (conservation of angular momentum)
- The red ω vector precesses around L, tracing out complex patterns
- The separation between ω and L reveals non-trivial dynamics
- The red marker dot shows how material points on the body trace paths through space

**Body-Centered Frame (middle):**
- Both ω and L vectors evolve in body coordinates
- For stable rotation (axes 1 or 3): ω stays nearly aligned with one principal axis
- For unstable rotation (axis 2): ω undergoes dramatic tumbling, displaying the tennis-racket effect
- The red marker dot stays fixed, helping you confirm you're riding with the body

**Poinsot Ellipsoids (bottom - optional, shown with --poinsot):**
- Shows the beautiful geometric interpretation of torque-free rotation in body-frame ω-space
- Green ellipsoid: All ω_body states with the same angular momentum magnitude |L|
- Orange (transparent) ellipsoid: All ω_body states with the same rotational energy E
- Yellow intersection curve: The only allowed path for ω_body(t) given both constraints
- Red trail: The actual trajectory of ω_body as it traces the intersection curve
- The ellipsoids are **fixed** in the body frame (they don't rotate with the body)
- For unstable rotation about axis 2, ω_body makes large excursions around the curve
- For stable rotation, ω_body stays near one of the intersection curve's stationary points

**Key Observations:**
- Rotation about smallest (I₁) or largest (I₃) inertia axes is stable
- Rotation about the intermediate axis (I₂) is unstable, causing dramatic tumbling
- The red trails make precession patterns visible in all three frames
- The Poinsot construction shows that ω is constrained to move on a 1D curve in 3D ω-space
- Energy and |L| are conserved (verified numerically with re-orthonormalization)

## Testing and Verification

The codebase includes a comprehensive test suite (`test_rigid_rotor.py`) with 24 tests that verify:

**Physics Algorithms:**
- `euler_rhs()`: Torque-free Euler equations with pure spin tests
- `rhs_full()`: Complete ODE system with axis evolution verification
- `body_to_space()`: Coordinate transformation correctness
- `calculate_dims_from_I()`: Moment of inertia calculations

**Conservation Laws:**
- Energy conservation (E = ½ Σ Iᵢ ωᵢ²) with <0.01% drift over tmax=10
- Angular momentum magnitude conservation
- Angular momentum direction conservation in space frame

**Numerical Properties:**
- Rotation matrix orthogonality (R^T R = I)
- Determinant preservation (det(R) = 1)
- Right-handedness (e₁ × e₂ = e₃)
- Long-term integration stability

**Physics Verification:**
- Stable rotation about axes 1 and 3
- Unstable rotation about axis 2 (tennis racket effect)
- Spherical top (I₁=I₂=I₃) dynamics
- Axially symmetric top (I₁=I₂≠I₃) precession

**Running Tests:**
```bash
# Run all tests locally
pytest test_rigid_rotor.py -v

# Run with coverage
pytest test_rigid_rotor.py --cov=rigid_rotor --cov-report=term-missing

# Run specific test category
pytest test_rigid_rotor.py -v -k "conservation"
```

**CI/CD:**
Tests run automatically via GitHub Actions (`.github/workflows/pytest.yml`) on:
- Every push to main
- Every pull request
- Daily at 00:00 UTC (scheduled)

The workflow tests against Python 3.8, 3.9, 3.10, and 3.11.

**Test Tolerances:**
- TIGHT_TOL = 1e-10: Exact mathematical relationships
- LOOSE_TOL = 1e-6: Numerical properties during integration
- INTEGRATION_TOL = 1e-4: Conservation laws over extended time

See `TESTING.md` for detailed documentation of all tests and their physical interpretations.
