import numpy as np
import sympy as sp
from .numeric import euler, rk2, rk4


class HomogeneousProjector:
    """
    Homogeneous Projector - preserves multiple invariants using nonlinear gradient flows.
    
    Supports two modes:
    1. Symbolic: invariants are sympy expressions, Jacobian computed symbolically
    2. Numerical: invariants are callable functions, Jacobian computed via finite differences
    """
    def __init__(self, invariants, initial_state, variables=None, gradients=None, max_iterations=2, tolerance=1e-12, integrator='euler', verbose=False, fd_epsilon=1e-8, **kwargs):
        """
        Initialize the HomogeneousProjector.
        
        Parameters:
        -----------
        invariants : sympy expression, list of sympy expressions, callable, or list of callables
            The invariants to preserve. If symbolic (sympy), variables must be provided.
            If callable, should accept state as separate arguments: f(x1, x2, ..., xn)
        variables : list of sympy symbols, optional
            The state variables [x1, ..., xn]. Required if invariants are symbolic.
        initial_state : array_like
            Initial state to compute target invariant values
        gradients : callable or list of callables, optional
            Analytical gradient functions. If provided, these are used instead of numerical
            finite differences or symbolic differentiation. Each gradient should be a callable
            that accepts state as separate arguments: grad_i(x1, x2, ..., xn) -> ndarray
            Must match the number and order of invariants.
        max_iterations : int, optional
            Maximum number of iterations for the correction (default: 2)
        tolerance : float, optional
            Convergence tolerance for corrections (default: 1e-12)
        integrator : str, optional
            Integration method for correction flows ('euler', 'rk2', 'rk4') (default: 'euler')
        verbose : bool, optional
            Print convergence information (default: False)
        fd_epsilon : float, optional
            Step size for finite difference Jacobian computation (default: 1e-8)
            Only used if gradients are not provided and invariants are callable.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.fd_epsilon = fd_epsilon
        self.gradients = gradients
        
        # Setup integrator
        if integrator in ['euler', 'rk1', 'rk2', 'rk4']:
            if integrator == 'rk1':
                integrator = 'euler'
            self.integrator = globals()[integrator]
            self.integrator_name = integrator
        else:
            raise ValueError(f"Unknown integrator: {integrator}. Available: 'euler'/'rk1', 'rk2', 'rk4'")

        # Convert single invariant to list for consistency
        if not isinstance(invariants, list):
            invariants = [invariants]
        self.invariants = invariants
        self.variables = variables
        
        # Convert single gradient to list if provided
        if gradients is not None:
            if not isinstance(gradients, list):
                gradients = [gradients]
            if len(gradients) != len(invariants):
                raise ValueError(f"Number of gradients ({len(gradients)}) must match number of invariants ({len(invariants)})")
            self.gradients = gradients
        
        # Determine mode and setup Jacobian
        if self.gradients is not None:
            # User-provided analytical gradients (fastest)
            self.J_fun = self._setup_gradient_mode()
            self.use_symbolic = False
        elif self._is_symbolic(invariants[0]):
            # Symbolic mode (compute gradients from sympy)
            if variables is None:
                raise ValueError("variables must be provided when using symbolic invariants")
            self.use_symbolic = True
            self.J_fun = self._setup_symbolic_mode()
        else:
            # Numerical mode (finite differences - slowest)
            self.use_symbolic = False
            self.J_fun = self._setup_numerical_mode()

        # Compute initial invariant values
        initial_state = np.asarray(initial_state, dtype=float).reshape(-1)
        self.H_initial = np.asarray(self.H_vec_fun(*initial_state), dtype=float).reshape(-1).copy()
    
    def _is_symbolic(self, invariant):
        """Check if an invariant is a sympy expression."""
        return isinstance(invariant, (sp.Expr, sp.Matrix))
    
    def _setup_gradient_mode(self):
        """Setup evaluators using user-provided analytical gradients."""
        # Wrap invariants to return vectorized output
        def H_vec_wrapper(*args):
            return np.array([inv(*args) for inv in self.invariants])
        self.H_vec_fun = H_vec_wrapper
        
        # Create Jacobian function from gradient functions
        def jacobian_from_gradients(*x_args):
            """
            Compute Jacobian from analytical gradient functions.
            
            Returns:
            --------
            J : ndarray, shape (m, n)
                Jacobian matrix where J[i,j] = ∂H_i/∂x_j
            """
            # Each gradient function returns a vector of shape (n,)
            # Stack them to get J of shape (m, n)
            grads = [grad(*x_args) for grad in self.gradients]
            return np.array(grads)
        
        return jacobian_from_gradients
    
    def _setup_symbolic_mode(self):
        """Setup evaluators for symbolic invariants."""
        H_vec_sym = sp.Matrix(self.invariants)               # shape (m, 1)
        J_sym = H_vec_sym.jacobian(self.variables)           # shape (m, n)
        self.H_vec_fun = sp.lambdify(self.variables, H_vec_sym, "numpy")
        return self._jacobian_symbolic(J_sym)
    
    def _setup_numerical_mode(self):
        """Setup evaluators for numerical (callable) invariants."""
        # Wrap invariants to return vectorized output
        def H_vec_wrapper(*args):
            return np.array([inv(*args) for inv in self.invariants])
        self.H_vec_fun = H_vec_wrapper
        return self._jacobian_numerical
    
    def _jacobian_symbolic(self, J_sym):
        """Create Jacobian evaluator from symbolic expression."""
        J_lambdified = sp.lambdify(self.variables, J_sym, "numpy")
        return J_lambdified
    
    def _jacobian_numerical(self, *x_args):
        """Compute Jacobian using finite differences.
        
        Parameters:
        -----------
        *x_args : unpacked state variables
        
        Returns:
        --------
        J : ndarray, shape (m, n)
            Jacobian matrix where J[i,j] = ∂H_i/∂x_j
        """
        x = np.asarray(x_args, dtype=float).reshape(-1)
        n = len(x)
        m = len(self.invariants)
        J = np.zeros((m, n))
        
        # Compute base function values
        H_base = np.array([inv(*x) for inv in self.invariants])
        
        # Compute finite differences for each component
        for j in range(n):
            x_perturbed = x.copy()
            x_perturbed[j] += self.fd_epsilon
            H_perturbed = np.array([inv(*x_perturbed) for inv in self.invariants])
            J[:, j] = (H_perturbed - H_base) / self.fd_epsilon
        
        return J
    
    def project(self, x):
        """Apply invariant projection to a state."""
        t_star = 1.0 
        x_tilde = np.asarray(x, dtype=float).reshape(-1)
        target = np.asarray(self.H_initial, dtype=float).reshape(-1)       # (m,)

        cur = np.asarray(self.H_vec_fun(*x_tilde), dtype=float).reshape(-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = (np.abs(cur) > self.tolerance) & (np.abs(target) > self.tolerance)
            s = np.zeros_like(cur)
            s[mask] = np.log(target[mask] / cur[mask])
        if np.linalg.norm(s, ord=np.inf) < self.tolerance:
            return x_tilde

        c = s / t_star
        x_corr = self._integration_step(x_tilde, c, t_star)
        
        converged = False
        for iteration in range(1, self.max_iterations):
            cur = np.asarray(self.H_vec_fun(*x_corr), dtype=float).reshape(-1)
            with np.errstate(divide='ignore', invalid='ignore'):
                s_new = np.zeros_like(cur)
                s_mask = (np.abs(cur) > self.tolerance) & (np.abs(target) > self.tolerance)
                s_new[s_mask] = np.log(target[s_mask] / cur[s_mask])
                rel = np.zeros_like(cur)
                r_mask = np.abs(target) > self.tolerance
                rel[r_mask] = np.abs(1.0 - cur[r_mask] / target[r_mask])
            if max(np.linalg.norm(s_new, ord=np.inf), np.max(rel)) < self.tolerance:
                converged = True
                break
            c_new = s_new / t_star
            x_corr = self._integration_step(x_corr, c_new, t_star)

        if not converged and self.verbose:
            print(f"HomogeneousProjector failed to converge after {self.max_iterations} iterations. "
                        f"Final error: {max(np.linalg.norm(s_new, ord=np.inf), np.max(rel)):.2e}, "
                        f"tolerance: {self.tolerance:.2e}")

        return x_corr
            
    def _compute_gradient_frame(self, x):
        """Compute the gradient frame G(x) = [∇H_1(x), ..., ∇H_m(x)]^T"""
        x = np.asarray(x, dtype=float).reshape(-1)
        J = np.asarray(self.J_fun(*x), dtype=float)   # shape (m, n)
        return J.T                                    # shape (n, m)
    
    def _solve_normal_equations(self, G, v):
        """Solve G @ λ = v using Tikhonov regularization"""
        GTG = G.T @ G  
        # Tikhonov regularization with fixed parameter
        reg = 1e-14
        try:
            lam = np.linalg.solve(GTG + reg * np.eye(GTG.shape[0]), v)
        except np.linalg.LinAlgError:
            lam, _, _, _ = np.linalg.lstsq(GTG, v, rcond=1e-15)
        return G @ lam                                # (n,)

    def _compute_vector_field(self, x, c):
        """Compute the combined vector field g(x) = U(x) * (c ⊙ H(x))"""
        x = np.asarray(x, dtype=float).reshape(-1)
        H_vals = np.asarray(self.H_vec_fun(*x), dtype=float).reshape(-1)   # (m,)
        G = self._compute_gradient_frame(x)                                # (n, m)
        v = c * H_vals                                                     # (m,)
        return self._solve_normal_equations(G, v)

    def _integration_step(self, x, c, t):
        """Apply integration step using the configured integrator."""
        def vector_field_func(x_state):
            return self._compute_vector_field(x_state, c)
        
        return self.integrator(x, t, vector_field_func)



        