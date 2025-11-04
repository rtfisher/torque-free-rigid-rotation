# Test Suite for Rigid Rotor Simulator

This document describes the verification test suite for `rigid_rotor.py`, which validates the physical correctness and numerical stability of the torque-free rigid body rotation simulator.

## Overview

The test suite (`test_rigid_rotor.py`) provides comprehensive verification of:
- **Physics algorithms**: Euler's equations and axis evolution
- **Conservation laws**: Energy, angular momentum magnitude and direction
- **Numerical properties**: Rotation matrix orthogonality, integration stability
- **Special cases**: Stable/unstable rotation axes, spherical and axially symmetric tops
- **Edge cases**: Degenerate moments of inertia

## Running the Tests

### Local Testing

```bash
# Run all tests with verbose output
pytest test_rigid_rotor.py -v

# Run with coverage report
pytest test_rigid_rotor.py --cov=rigid_rotor --cov-report=term-missing

# Run specific test category
pytest test_rigid_rotor.py -v -k "conservation"
pytest test_rigid_rotor.py -v -k "euler_rhs"
```

### CI/CD Testing

Tests run automatically via GitHub Actions on:
- Every push to `main` branch
- Every pull request
- Daily at 00:00 UTC (scheduled)

The workflow tests against Python 3.8, 3.9, 3.10, and 3.11 to ensure compatibility.

## Test Categories

### 1. Euler Equations Tests (`test_euler_rhs_*`)

**Purpose**: Verify that Euler's torque-free equations are correctly implemented.

**Tests**:
- `test_euler_rhs_pure_spin_axis1/2/3`: Pure rotation about any principal axis should produce zero angular acceleration
- `test_euler_rhs_symmetry`: Verify the mathematical form of Euler's equations
- `test_euler_rhs_zero_omega`: Zero angular velocity should produce zero acceleration

**Physics Verified**:
```
dω₁/dt = ((I₂ - I₃)/I₁) ω₂ ω₃
dω₂/dt = ((I₃ - I₁)/I₂) ω₃ ω₁
dω₃/dt = ((I₁ - I₂)/I₃) ω₁ ω₂
```

### 2. Full ODE System Tests (`test_rhs_full_*`)

**Purpose**: Verify the complete 12-dimensional ODE system that evolves both ω and the body axes.

**Tests**:
- `test_rhs_full_axis_evolution_formula`: Verify de/dt = ω_space × e
- `test_rhs_full_pure_rotation`: Pure principal axis rotation should keep ω constant

**Physics Verified**: Rigid body kinematics for axis vectors in space frame.

### 3. Coordinate Transform Tests (`test_body_to_space_*`)

**Purpose**: Ensure transformations between body and space frames are correct.

**Tests**:
- `test_body_to_space_identity`: Identity rotation leaves vectors unchanged
- `test_body_to_space_preserves_lengths`: Rotations preserve vector lengths (isometry)

**Physics Verified**: R is a proper orthogonal transformation.

### 4. Moment of Inertia Tests (`test_calculate_dims_*`)

**Purpose**: Verify the calculation of box dimensions from moments of inertia.

**Tests**:
- `test_calculate_dims_triangle_inequality`: Reject unphysical moments
- `test_calculate_dims_forward_backward`: Calculated dims reproduce original I values
- `test_calculate_dims_positive`: All dimensions are positive

**Physics Verified**:
```
For uniform box: I₁ ∝ (b² + c²), I₂ ∝ (a² + c²), I₃ ∝ (a² + b²)
Triangle inequality: I₁ + I₂ > I₃, etc.
```

### 5. Rotation Matrix Property Tests (`test_rotation_matrix_*`)

**Purpose**: Ensure numerical integration maintains rotation matrix properties.

**Tests**:
- `test_rotation_matrix_orthogonality_short_integration`: R^T R = I and det(R) = 1
- `test_rotation_matrix_right_handed`: e₁ × e₂ = e₃

**Physics Verified**: R remains in SO(3) despite numerical integration.

### 6. Conservation Law Tests (`test_*_conservation`)

**Purpose**: Verify fundamental conservation laws of torque-free motion.

**Tests**:
- `test_energy_conservation`: Rotational kinetic energy E = ½ Σ Iᵢ ωᵢ² is conserved
- `test_angular_momentum_magnitude_conservation`: |L| is conserved
- `test_angular_momentum_direction_conservation`: L_space is fixed in inertial frame

**Tolerance**: < 0.01% drift over integration time (rtol=1e-4)

**Physics Verified**: For torque-free motion, E and L are constants of motion.

### 7. Stability Tests (`test_*_rotation_axis*`)

**Purpose**: Verify the classic stability results for rigid body rotation.

**Tests**:
- `test_stable_rotation_axis1`: Rotation about smallest I is stable
- `test_stable_rotation_axis3`: Rotation about largest I is stable
- `test_unstable_rotation_axis2`: Rotation about intermediate I is unstable (tennis racket effect)

**Physics Verified**:
- Axes with extremal moments of inertia are stable
- Intermediate axis exhibits exponential instability

### 8. Long Integration Tests (`test_long_integration_conservation`)

**Purpose**: Ensure numerical stability over extended time periods.

**Tests**:
- `test_long_integration_conservation`: Conservation laws hold for tmax=10.0 with <1% drift

**Numerical Methods Verified**:
- `scipy.integrate.solve_ivp` with rtol=1e-10, atol=1e-10
- Gram-Schmidt re-orthonormalization prevents rotation matrix drift

### 9. Edge Case Tests (`test_spherical_top`, `test_axially_symmetric_top`)

**Purpose**: Test degenerate/special inertia configurations.

**Tests**:
- `test_spherical_top`: I₁ = I₂ = I₃ → ω_body constant (free rotation)
- `test_axially_symmetric_top`: I₁ = I₂ ≠ I₃ → regular precession

**Physics Verified**: Special symmetries produce expected simplified dynamics.

### 10. Performance Tests (`test_simulation_completes_quickly`)

**Purpose**: Regression testing for computational performance.

**Tests**:
- `test_simulation_completes_quickly`: 10s simulation at 60fps completes in <5s

## Test Tolerances

The test suite uses three tolerance levels:

- **TIGHT_TOL** = 1e-10: For exact mathematical relationships (pure spin, zero cases)
- **LOOSE_TOL** = 1e-6: For numerical properties maintained during integration
- **INTEGRATION_TOL** = 1e-4: For conservation laws over extended time

These reflect realistic expectations for floating-point arithmetic and numerical ODE solvers.

## Key Verification Points

### Single-Timestep Tests
Many tests verify single evaluations of `euler_rhs()` and `rhs_full()` without integration. These are **fast** (<0.01s each) and catch:
- Implementation errors in physics equations
- Sign errors, indexing errors
- Coordinate frame confusion

### Short Integration Tests
Tests using `tmax=0.1` to `tmax=2.0` verify:
- Numerical stability over short periods
- Conservation law accuracy
- Rotation matrix orthogonality maintenance

### Long Integration Tests
Tests using `tmax=10.0` verify:
- Long-term conservation properties
- Absence of catastrophic drift
- Gram-Schmidt re-orthonormalization effectiveness

## Expected Test Results

All 26 tests should **PASS** on a correctly functioning installation. Typical execution time on modern hardware: **10-15 seconds** for the full suite.

Example output:
```
test_rigid_rotor.py::test_euler_rhs_pure_spin_axis1 PASSED            [  3%]
test_rigid_rotor.py::test_euler_rhs_pure_spin_axis2 PASSED            [  7%]
...
test_rigid_rotor.py::test_simulation_completes_quickly PASSED         [100%]

======================== 26 passed in 12.34s =========================
```

## Interpreting Failures

### Physics Equation Errors
If `test_euler_rhs_*` or `test_rhs_full_*` fail → Check implementation of Euler's equations and axis evolution formulas

### Conservation Violations
If `test_*_conservation` fail → Check integration tolerances or re-orthonormalization

### Numerical Instability
If `test_rotation_matrix_*` fail → Rotation matrices drifting from SO(3), check re-orthonormalization

### Performance Regression
If `test_simulation_completes_quickly` fails → Check for inefficient code changes

## Adding New Tests

When adding features, consider adding tests for:
1. **Mathematical correctness**: Does the algorithm implement the physics correctly?
2. **Conservation laws**: Are the appropriate quantities conserved?
3. **Edge cases**: How does it behave with degenerate inputs?
4. **Performance**: Does it maintain acceptable speed?

## References

The physics verified by these tests is based on:
- Goldstein, *Classical Mechanics*, Chapter 5 (Rigid Body Motion)
- Landau & Lifshitz, *Mechanics*, §37 (Free rotation of a rigid body)
- The tennis racket instability (intermediate axis theorem)

## CI Badge

Add this badge to your README.md to show test status:

```markdown
![Tests](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/pytest.yml/badge.svg)
```
