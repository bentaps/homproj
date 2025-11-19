import numpy as np
import sympy as sp
import scipy
from . import numeric

class Projector:
    """
    Projector - preserves invariants using projection onto level sets.
    """
    def __init__(self, invariants, variables, initial_state, max_iterations=50, 
                 tolerance=1e-14, relaxation=1.0, verbose=1, **kwargs):
        """
        Initialize the Projector.

        Parameters:
        -----------
        invariants : sympy expression or list of sympy expressions
            The invariants to preserve
        variables : list of sympy symbols
            The state variables [x1, ..., xn]
        initial_state : array_like
            Initial state to compute target invariant values
        max_iterations : int, optional
            Maximum number of iterations for the correction (default: 50)
        tolerance : float, optional
            Convergence tolerance for corrections (default: 1e-14)
        relaxation : float, optional
            Relaxation parameter for the projection (default: 1.0)
        """
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.relaxation = relaxation
        self.variables = variables
        self.error_msg_count = 0  # Count of error messages printed
        
        # Convert single invariant to list for consistency
        if not isinstance(invariants, list):
            invariants = [invariants]
        self.invariants = invariants
        
        # Create lambdified functions for all invariants
        self.H_funs = [sp.lambdify(variables, H, "numpy") for H in invariants]
        self.H_fun = self.H_funs[0]  # Primary function for compatibility
        
        # Create gradient functions for all invariants
        self.gradH_funs = []
        for H in invariants:
            gradinvariants = [sp.diff(H, var) for var in variables]
            self.gradH_funs.append(sp.lambdify(variables, gradinvariants, "numpy"))
        
        # Compute initial invariant values
        self.H_initial = [H_fun(*initial_state) for H_fun in self.H_funs]
    
    

    
    def project(self, y):
        """
        Apply invariant projection to a state.
        
        Parameters:
        -----------
        y : array_like
            State to project
            
        Returns:
        --------
        array_like
            Projected state
        """
        return self._project_to_invariants(y, self.H_initial)
    
    def _project_to_invariants(self, x_tilde, H_targets):
        """
        Project state x_tilde to satisfy H_i(x) = H_targets[i] for all i.
        
        Uses simultaneous linear projection onto all invariant manifolds.
        This implements the Newton linearization approach:
        δx = J^T(JJ^T)^(-1)(c - H(x))
        where J is the Jacobian with rows ∇H_i(x)^T.
        """
        x = x_tilde.astype(float).copy()
        atol = self.tolerance
        
        for _ in range(self.max_iterations):
            # Compute current invariant values
            H_vals = np.array([H_fun(*x) for H_fun in self.H_funs], dtype=float)
            
            # Compute residual r = c - H(x)
            r = np.array(H_targets, dtype=float) - H_vals
            
            # Check convergence using scaled residual norm
            scaled_residual = np.where(np.abs(H_targets) > 1e-14, 
                                     np.abs(r) / np.abs(H_targets), 
                                     np.abs(r))
            max_scaled_residual = np.max(scaled_residual)
            
            if max_scaled_residual <= atol:
                break
            
            # Build Jacobian J with rows grad H_i(x)
            J = np.vstack([np.array(gradH_fun(*x), dtype=float).reshape(-1) 
                          for gradH_fun in self.gradH_funs])  # q×d
            
            # Compute Gram matrix G = JJ^T (q×q)
            G = J @ J.T
            
            try:
                lam = np.linalg.solve(G, r)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if lin solve fails
                lam = np.linalg.lstsq(G, r, rcond=1e-15)[0]
            
            # Compute correction δx = J^T * λ
            dx = J.T @ lam
            
            # Apply correction with relaxation
            x = x + self.relaxation * dx
        
        else:
            # Loop completed without convergence
            if self.verbose:
                if self.verbose==1 and self.error_msg_count == 1:
                    return x
                H_vals_final = np.array([H_fun(*x) for H_fun in self.H_funs], dtype=float)
                r_final = np.array(H_targets, dtype=float) - H_vals_final
                scaled_residual_final = np.where(np.abs(H_targets) > 1e-14,
                                               np.abs(r_final) / np.abs(H_targets),
                                               np.abs(r_final))
                max_final_error = np.max(scaled_residual_final)
                print(f"Warning: ProjectionWrapper failed to achieve tolerance {self.tolerance:.2e}. "
                      f"Maximum scaled residual: {max_final_error:.2e} after {self.max_iterations} iterations.")
                self.error_msg_count += 1
        
        return x