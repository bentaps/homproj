"""
Verified high-order numerical integration methods.
Using only well-tested coefficients from authoritative sources.
"""

import numpy as np
import scipy.optimize

# ===== BASIC METHODS (VERIFIED) =====

def euler(x, h, f):
    """First-order Euler method."""
    f = _adapt_func_signature(f)
    return x + h * f(x)

def rk2(x, h, f):
    """Second-order Runge-Kutta (midpoint method)."""
    f = _adapt_func_signature(f)
    k1 = f(x)
    k2 = f(x + h/2 * k1)
    return x + h * k2

def rk4(x, h, f):
    """Fourth-order Runge-Kutta method (classical)."""
    f = _adapt_func_signature(f)
    k1 = f(x)
    k2 = f(x + h/2 * k1)
    k3 = f(x + h/2 * k2)
    k4 = f(x + h * k3)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)

# ===== SYMPLECTIC METHODS =====

def implicit_rk(x, h, f, a, b, c, max_iter=50, tol=None):
    """
    General implicit Runge-Kutta method for solving ODEs using a Butcher tableau.
    
    Parameters:
    -----------
    x : array_like
        Current state vector.
    h : float
        Step size.
    f : callable
        Function that computes the derivative of x with respect to time.
        Can accept either f(x) or f(t, x).
    a : array_like
        Runge-Kutta matrix (coefficients for stages).
    b : array_like
        Weights for the stages in the final update.
    c : array_like
        Nodes (time points for stages).
    max_iter : int, optional
        Maximum number of iterations for the fixed-point method. Default is 10.
    tol : float, optional
        Tolerance for convergence of fixed-point iterations. If None, will be set
        to 1e-12 * max(abs(x)).
        
    Returns:
    --------
    array_like
        The estimated solution at time t + h.
    """
    # Adapt function signature if needed
    f = _adapt_func_signature(f)
    
    # Number of stages
    s = len(b)
    
    # State dimension
    n = len(x)
    
    # Set tolerance if not provided
    if tol is None:
        tol = 1e-14 * np.max(np.abs(x))
    
    # Initialize stage values
    X = np.zeros((s, n))
    
    # start with K^0 = f(t,x)
    K = np.tile(f( x), (s, 1))
    for _ in range(max_iter):
        K_old = K.copy()
        for i in range(s):
            Xi = x + h * (a[i, :] @ K_old)
            K[i] = f(Xi)
        if np.linalg.norm(K - K_old, ord=np.inf) < tol:
            break
    else:
        print("Warning: stage iteration did not converge")

    x_next = x + h * (b @ K)
    return x_next

def gll4(x, h, f):
    """
    Fourth-order symplectic Gauss-Legendre method.
    
    This is a 4th order symplectic integrator specifically designed for Hamiltonian systems.
    It preserves the symplectic structure of phase space and provides excellent long-term
    energy conservation compared to non-symplectic methods of the same order.
    
    Parameters:
    -----------
    x : array_like
        Current state vector.
    h : float
        Step size.
    f : callable
        Function that computes the derivative of x with respect to time.
        Can accept either f(x) or f(t, x).
        
    Returns:
    --------
    array_like
        The estimated solution at time t + h.
        
    References:
    -----------
    - Hairer, Ernst, Christian Lubich, and Gerhard Wanner. "Geometric Numerical Integration:
      Structure-Preserving Algorithms for Ordinary Differential Equations." Springer, 2006.
    """
    # Define the Butcher tableau for GLL4 (2-stage Gauss method)
    sqrt3 = np.sqrt(3)
    a = np.array([
        [1/4, 1/4 - sqrt3/6],
        [1/4 + sqrt3/6, 1/4]
    ])
    
    b = np.array([1/2, 1/2])
    c = np.array([1/2 - sqrt3/6, 1/2 + sqrt3/6])
    
    # Use the general implicit RK solver
    return implicit_rk(x, h, f, a, b, c)

def midpoint2(x, h, f):
    """Second-order symplectic midpoint method."""
    # Midpoint method is a special case of implicit RK with one stage
    a = np.array([[0.5]])
    b = np.array([1.0])
    c = np.array([0.5])
    return implicit_rk(x, h, f, a, b, c)


def pseudo_symplectic(x, h, f):
    # Order 4, symplectic order 9, RK method
    a = 1/(2 - 2**(1/3))
    a1, a2, a3 = a, 1 - 2*a, a
    
    k1 = f(x)
    k2 = f(x + h * a1/2 * k1)
    k3 = f(x + h * a1 * k2)
    k4 = f(x + h * (a1/2 * k1 + (a1/2 + a2/2) * k3))
    k5 = f(x + h * (a1 * k2 + a2 * k4))
    k6 = f(x + h * (a1/2 * k1 + (a1/2 + a2/2) * k3 + (a2/2 + a3/2) * k5))
    k7 = f(x + h * (a1 * k2 + a2 * k4 + a3 * k6))
    
    return x + h * (a1/4 * k1 + a1/2 * k2 + (a1 + a2)/4 * k3 + 
                    a2/2 * k4 + (a2 + a3)/4 * k5 + a3/2 * k6 + a3/4 * k7)

def gll6(x, h, f):
    # Define the Butcher tableau for GLL6 (3-stage Gauss method)
    a = np.array([
        [5/36,            2/9 - np.sqrt(15)/15,  5/36 - np.sqrt(15)/30],
        [5/36 + np.sqrt(15)/24,  2/9,            5/36 - np.sqrt(15)/24],
        [5/36 + np.sqrt(15)/30,  2/9 + np.sqrt(15)/15,  5/36]
    ], dtype=float)

    b = np.array([5/18, 4/9, 5/18], dtype=float)
    c = np.array([
        0.5 - np.sqrt(15)/10,
        0.5,
        0.5 + np.sqrt(15)/10
    ], dtype=float)
    
    # Use the general implicit RK solver
    return implicit_rk(x, h, f, a, b, c)


######################################### 
# Integration with fixed step size and scipy-compatible interface
#########################################

def _adapt_func_signature(f):
    """
    Check if function accepts one argument f(x) or two arguments f(t, x) and adapt accordingly.
    Returns a function that accepts only one argument x, with an optional keyword argument t.
    """
    import inspect
    sig = inspect.signature(f)
    
    if len(sig.parameters) == 1:
        # Function already has the form f(x)
        # Create a wrapper that ignores the t parameter
        def wrapped_f(x, t=0.0):
            return f(x)
        return wrapped_f
    elif len(sig.parameters) == 2:
        # Function has the form f(t, x), adapt to f(x, t=0.0)
        def wrapped_f(x, t=0.0):
            return f(t, x)
        return wrapped_f
    else:
        raise ValueError(f"Function must accept either 1 or 2 arguments, got {len(sig.parameters)}")

class FixedStepODESolution:
    """Class representing the solution of an ODE solved with a fixed step method."""
    
    def __init__(self, t, y, success=True, message=""):
        self.t = t
        self.y = y
        self.success = success
        self.message = message
        self.nfev = None # Number of function evaluations (approximate)
        self.njev = 0  # No Jacobian evaluations
        self.nlu = 0  # No LU decompositions
        self.status = 0 if success else -1
        
    def __call__(self, t):
        """Evaluate solution at time t using linear interpolation."""
        if np.isscalar(t):
            # Find closest indices
            idx = np.searchsorted(self.t, t)
            if idx == 0:
                return self.y[0]
            elif idx >= len(self.t):
                return self.y[-1]
            else:
                # Linear interpolation
                t0, t1 = self.t[idx-1], self.t[idx]
                y0, y1 = self.y[idx-1], self.y[idx]
                return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
        else:
            # Handle array of time values
            t = np.asarray(t)
            result = np.zeros((len(t), self.y.shape[1]))
            for i, ti in enumerate(t):
                result[i] = self(ti)
            return result

def solve_ivp_fixed_step(f, t_span, y0, method='rk4', h=0.01, dense_output=False):
    """
    Solve an initial value problem using a fixed-step method.
    
    This function provides a compatible interface with scipy's solve_ivp,
    but uses fixed step size methods.
    
    Parameters:
    -----------
    f : callable
        Right-hand side of the ODE system. The calling signature is f(t, y) or f(y).
        Here t is a scalar and y is a 1D array.
    t_span : 2-tuple
        Interval of integration (t0, tf).
    y0 : array_like
        Initial state.
    method : callable or str, optional
        Integration method. Can be a callable with signature method(x, h, f) or
        a string referring to a method in the METHODS registry.
        Default is 'rk4'.
    h : float, optional
        Fixed step size. Default is 0.01.
    dense_output : bool, optional
        Whether to compute a continuous solution. Currently ignored as all
        solutions use linear interpolation. Default is False.
        
    Returns:
    --------
    solution : FixedStepODESolution
        Object with the solution data (compatible with scipy's OdeSolution).
        Important attributes:
        - t : array
            Time points.
        - y : ndarray
            Values of the solution at t (shape is (len(t), len(y0))).
        - success : bool
            True if the integration succeeded.
        - message : str
            Description of the cause of the termination.
        The solution object can be called to get values at arbitrary time points.
    """
    # Get the method function if a string is provided
    if isinstance(method, str):
        if method in METHODS:
            method_func = METHODS[method]['function']
        else:
            available = list(METHODS.keys())
            raise ValueError(f"Method '{method}' not available. Choose from: {available}")
    else:
        # Assume method is a callable
        method_func = method
    
    # Handle different function signatures
    adapted_f = _adapt_func_signature(f)
    
    t0, tf = t_span
    n_steps = int(np.ceil((tf - t0) / h))
    actual_h = (tf - t0) / n_steps  # Adjust step size to hit tf exactly
    
    t_values = np.linspace(t0, tf, n_steps + 1)
    y0 = np.asarray(y0)
    y_values = np.zeros((n_steps + 1, len(y0)))
    y_values[0] = y0
    
    success = True
    message = "Integration successful."
    
    try:
        for i in range(n_steps):
            current_t = t_values[i]
            current_y = y_values[i]
            
            # Create a time-dependent version of f for this step
            def f_step(y):
                return adapted_f(y, t=current_t)
            
            # Perform one step of integration
            y_values[i + 1] = method_func(current_y, actual_h, f=f_step)
    except Exception as e:
        success = False
        message = f"Integration failed: {str(e)}"
    
    # Reshape to match scipy.integrate.solve_ivp output
    y_values = y_values.T
    
    # Return solution object with scipy-compatible interface
    return FixedStepODESolution(t_values, y_values, success, message)


# ===== METHOD REGISTRY =====

# Dictionary of available methods with their properties
METHODS = {
    'euler': {
        'function': euler,
        'order': 1,
        'stages': 1,
        'type': 'explicit',
        'symplectic': False
    },
    'rk2': {
        'function': rk2,
        'order': 2,
        'stages': 2,
        'type': 'explicit',
        'symplectic': False
    },
    'rk4': {
        'function': rk4,
        'order': 4,
        'stages': 4,
        'type': 'explicit',
        'symplectic': False
    },
    'gll4': {
        'function': gll4,
        'order': 4,
        'stages': 2,
        'type': 'implicit',
        'symplectic': True
    },
    'gll6': {
        'function': gll6,
        'order': 6,
        'stages': 3,
        'type': 'implicit',
        'symplectic': True
    }
}