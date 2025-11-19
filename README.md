# Invariant-Preserving Integration with `homproj`

A Python package for geometric integration that preserves invariants (conservation laws) to machine precision. Extends `scipy.integrate.solve_ivp` with automatic invariant preservation for general nonlinear invariants with minimal overhead.

See our [preprint](https://arxiv.org/pdf/2511.02131) for details! 

## Features

- **Exact invariant preservation** to high precision
- **Drop-in replacement** for `scipy.integrate.solve_ivp`
- **Adaptive integration** with DOP853, RK45, and other high-order methods
- **Minimal overhead** no methods require non-linear solves
- **Multiple invariants** preserved simultaneously

## Installation

```bash
# Clone the repository
git clone https://github.com/bentaps/homproj.git
cd invariant-preservation

# Install in development mode
pip install -e .
```

or using `pip`

```bash
pip install homproj
```

## Quick Start

The main interface is `homproj.solve_ivp`, which works like `scipy.integrate.solve_ivp` but with invariant preservation:

```python
import numpy as np
from homproj import solve_ivp

# Define your ODE
def kepler_dynamics(t, y):
    """Kepler problem: dy/dt = f(y) where y = [q1, q2, p1, p2]"""
    q1, q2, p1, p2 = y
    r = np.sqrt(q1**2 + q2**2)
    r3 = r**3
    return np.array([p1, p2, -q1/r3, -q2/r3])

y0 = np.array([1.0, 0.0, 0.0, 1.0])  # Initial conditions

# Standard integration (no projection) - energy drifts!
from scipy.integrate import solve_ivp as scipy_solve_ivp
sol_standard = scipy_solve_ivp(kepler_dynamics, (0, 100), y0, method='DOP853')
print(f"Energy drift (standard): ~1e-8")

# With invariant preservation - energy conserved to machine precision!
sol = solve_ivp(
    fun=kepler_dynamics,
    t_span=(0, 100),
    y0=y0,
    method='DOP853'
    # ... add invariants below
)
```

## Usage

### Option 1: Symbolic Invariants

The simplest approach, just provide symbolic expressions and gradients are done for you:

```python
import sympy as sp
from homproj import solve_ivp

# Define symbolic variables
q1, q2, p1, p2 = sp.symbols('q1 q2 p1 p2', real=True)
r = sp.sqrt(q1**2 + q2**2)

# Define invariants symbolically
H = sp.Rational(1,2) * (p1**2 + p2**2) - 1/r  # Energy
L = q1*p2 - q2*p1                              # Angular momentum

# Preserve one invariant
sol = solve_ivp(
    fun=kepler_dynamics,
    t_span=(0, 100),
    y0=y0,
    method='DOP853',
    invariants=[H],            # Single invariant
    variables=[q1, q2, p1, p2]
)

# Or preserve multiple invariants
sol = solve_ivp(
    fun=kepler_dynamics,
    t_span=(0, 100),
    y0=y0,
    method='DOP853',
    invariants=[H, L],         # Multiple invariants
    variables=[q1, q2, p1, p2]
)

# Energy and momentum preserved to ~1e-15!
```

### Option 2: Numerical Functions

If you provide numpy functions, the gradients will be calculated using finite differences:


```python
# Define invariants as functions
def energy(q1, q2, p1, p2):
    return 0.5 * (p1**2 + p2**2) - 1.0/np.sqrt(q1**2 + q2**2)

def angular_momentum(q1, q2, p1, p2):
    return q1*p2 - q2*p1

sol = solve_ivp(
    fun=kepler_dynamics,
    t_span=(0, 100),
    y0=y0,
    method='DOP853',
    invariants=[energy, angular_momentum]
    # No 'variables' needed for numerical functions!
)
```

### Option 3: Functions + Gradients

Provide analytical gradients:

```python
def energy(q1, q2, p1, p2):
    r = np.sqrt(q1**2 + q2**2)
    return 0.5 * (p1**2 + p2**2) - 1.0/r

def grad_energy(q1, q2, p1, p4):
    r = np.sqrt(q1**2 + q2**2)
    r3 = r**3
    return np.array([q1/r3, q2/r3, p1, p2])

def angular_momentum(q1, q2, p1, p2):
    return q1*p2 - q2*p1

def grad_angular_momentum(q1, q2, p1, p2):
    return np.array([p2, -p1, -q2, q1])

sol = solve_ivp(
    fun=kepler_dynamics,
    t_span=(0, 100),
    y0=y0,
    method='DOP853',
    invariants=[energy, angular_momentum],
    gradients=[grad_energy, grad_angular_momentum]
)

```

## Key Parameters

```python
sol = solve_ivp(
    fun,                    # dy/dt = fun(t, y)
    t_span,                 # (t0, tf) integration interval
    y0,                     # Initial state
    method='DOP853',        # 'RK45', 'DOP853', 'Radau', 'BDF', etc. 
    rtol=1e-9,              # Relative tolerance for solution
    atol=1e-12,             # Absolute tolerance for solution
    invariants=[...],       # List of invariants (sympy or functions)
    variables=[...],        # List of sympy symbols (if using sympy)
    gradients=[...],        # Optional: analytical gradients
    integrator='rk2',       # Optional: integration method for projection
    itol=1e-14,             # Invariant preservation tolerance
    max_iterations=10,      # Projection iterations per step
    **kwargs,               # Optional args passed to scipy.integrate.solve_ivp
)
```

## Available Methods

- `RK45`: Explicit Runge-Kutta 4(5) [default]
- `DOP853`: Explicit Runge-Kutta 8(5,3) [recommended for high accuracy]
- `Radau`: Implicit Runge-Kutta (for stiff problems)
- `BDF`: Backward differentiation (for stiff problems)
- `RK23`: Explicit Runge-Kutta 2(3)

## Complete Example: Kepler Problem

See `simple_example.ipynb` for a complete tutorial showing:
- Standard integration vs. invariant-preserving integration
- Single invariant preservation (energy or angular momentum)
- Multiple invariant preservation (energy + momentum + Runge-Lenz vector)
- Comparison of fixed-step vs. adaptive methods
- Visualization of orbits and error growth

```python
# Quick example from simple_example.ipynb
import numpy as np
import sympy as sp
from homproj import solve_ivp

def kepler_dynamics(t, y):
    q1, q2, p1, p2 = y
    r = np.sqrt(q1**2 + q2**2)
    r3 = r**3
    return np.array([p1, p2, -q1/r3, -q2/r3])

q1, q2, p1, p2 = sp.symbols('q1 q2 p1 p2', real=True)
r = sp.sqrt(q1**2 + q2**2)

# Three conserved quantities in Kepler problem
H = sp.Rational(1,2) * (p1**2 + p2**2) - 1/r  # Energy
L = q1*p2 - q2*p1                              # Angular momentum
A = p2*L - q1/r                                # Runge-Lenz component

y0 = np.array([0.2, 0.0, 0.0, 4.358899])

sol = solve_ivp(
    fun=kepler_dynamics,
    t_span=(0, 100),
    y0=y0,
    method='DOP853',
    rtol=1e-9,
    atol=1e-12,
    invariants=[H, L, A],
    variables=[q1, q2, p1, p2],
    itol=1e-14
)

```

## Linear Homogeneous Projection

For special cases where invariants are homogeneous (e.g., $H(\lambda^{\nu} y) = \lambda^k H(y)$), you can use `LinearHomogeneousProjector` for maximum efficiency.

## Citation

If you use this code in research, please cite:

```bibtex
@article{tapley2025explicit,
  title={Explicit invariant-preserving integration of differential equations using homogeneous projection},
  author={Tapley, Benjamin Kwanen},
  journal={arXiv preprint arXiv:2511.02131},
  year={2025}
}
```

## More examples 

- **`simple_example.ipynb`**: Complete tutorial - **start here!**
- There are more examples for ODEs (Kepler and double pendulums) and PDEs (KdV and Camassa-Holm)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
