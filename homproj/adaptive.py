"""
Adaptive ODE solvers with customizable projection methods.

This module provides a generic framework for wrapping any SciPy ODE solver
with customizable projection/correction methods for invariant preservation
or other geometric integration techniques.
"""


import scipy
from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate._ivp import RK45, RK23, DOP853, Radau, BDF, LSODA
from homproj.homproj import HomogeneousProjector
from homproj.linhomproj import LinearHomogeneousProjector, AlternatingLinearHomogeneousProjector

METHOD_MAP = {
    'RK45': RK45, 'RK23': RK23, 'DOP853': DOP853,
    'Radau': Radau, 'BDF': BDF, 'LSODA': LSODA
}


class ProjectedOdeSolver(OdeSolver):
    """
    Generic ODE solver with customizable projection/correction methods.
    
    This class can wrap any ODE solver that inherits from OdeSolver and applies
    any projection or correction method after each step. The projection method
    is completely flexible and can be any callable that takes a state vector
    and returns a corrected state vector.
    """
    
    def __init__(self, solver_class, fun, t0, y0, t_bound, project, **kwargs):
        """
        Initialize the projected ODE solver.
        
        Parameters:
        -----------
        solver_class : class
            The ODE solver class to wrap (e.g., RK45, RK23, DOP853, Radau, BDF)
        fun : callable
            Right-hand side function
        t0 : float
            Initial time
        y0 : array_like
            Initial state
        t_bound : float
            Final time
        project : callable
            Projection/correction method that takes a state vector and returns
            a corrected state vector. Can be any callable with signature:
            corrected_y = project(y)
            Examples:
            - LHSWrapper.correct method
            - Custom projection functions
            - Lambda functions for simple corrections
        **kwargs
            Additional arguments passed to the underlying solver
        """
        # Initialize the base OdeSolver (this handles basic setup)
        super().__init__(fun, t0, y0, t_bound, kwargs.get('vectorized', False))
        
        # Create an instance of the underlying solver
        self.solver = solver_class(fun, t0, y0, t_bound, **kwargs)
        
        # Store the projection method
        self.project = project
        
        # Delegate key attributes to the underlying solver
        self._update_attributes()
        
    def _update_attributes(self):
        """Update attributes from the underlying solver."""
        # Copy important attributes from the underlying solver
        self.t = self.solver.t
        self.y = self.solver.y
        self.h_abs = getattr(self.solver, 'h_abs', None)
        self.nfev = self.solver.nfev
        self.njev = getattr(self.solver, 'njev', 0)
        self.nlu = getattr(self.solver, 'nlu', 0)
        self.status = self.solver.status
        
        # Copy step history for dense output
        if hasattr(self.solver, 't_old'):
            self.t_old = self.solver.t_old
        else:
            self.t_old = None
        if hasattr(self.solver, 'y_old'):
            self.y_old = self.solver.y_old
        else:
            self.y_old = None
        
        # Copy additional solver-specific attributes needed for dense output
        if hasattr(self.solver, 'K'):
            self.K = self.solver.K
        if hasattr(self.solver, 'h'):
            self.h = self.solver.h
        if hasattr(self.solver, 'f'):
            self.f = self.solver.f
        if hasattr(self.solver, 'fun'):
            self.fun = self.solver.fun
        if hasattr(self.solver, 'n_stages'):
            self.n_stages = self.solver.n_stages
        if hasattr(self.solver, 'A_EXTRA'):
            self.A_EXTRA = self.solver.A_EXTRA
        if hasattr(self.solver, 'C_EXTRA'):
            self.C_EXTRA = self.solver.C_EXTRA
        
    def _step_impl(self):
        """
        Override the step implementation to include projection/correction.
        
        This delegates to the underlying solver's step implementation and then
        applies the projection method if the step was successful.
        """
        # Perform the step using the underlying solver
        success, message = self.solver._step_impl()
        
        # Update our attributes from the underlying solver
        self._update_attributes()
        
        # Apply projection/correction if the step was successful
        if success:
            # Apply the projection method to the current state
            if hasattr(self.project, 'project'):
                # If it's an object with a project method (new interface)
                corrected_y = self.project.project(self.y)
            elif callable(self.project):
                # If it's a function
                corrected_y = self.project(self.y)
            else:
                raise ValueError("project must be callable or have a 'project' method")

            # Update both our state and the underlying solver's state
            self.y = corrected_y
            self.solver.y = corrected_y
        
        return success, message
    
    def _dense_output_impl(self):
        """
        Override dense output to delegate to the underlying solver.
        
        Note: The dense output uses the underlying solver's interpolation,
        which does NOT include projection. For exact invariant preservation
        at interpolated points, re-evaluate using the dense output followed
        by projection, or use t_eval to specify evaluation points.
        """
        return self.solver._dense_output_impl()


def create_projected_solver(solver_class, project_method):
    """
    Factory function to create a projected solver class for a specific base solver.
    """
    class SpecificProjectedSolver(ProjectedOdeSolver):
        def __init__(self, fun, t0, y0, t_bound, **kwargs):
            super().__init__(
                solver_class=solver_class,
                fun=fun, t0=t0, y0=y0, t_bound=t_bound,
                project=project_method,
                **kwargs
            )
    
    # Set a meaningful name for the class
    SpecificProjectedSolver.__name__ = f"Projected{solver_class.__name__}"
    return SpecificProjectedSolver


def solve_ivp(fun, t_span, y0, method='RK45', 
                      invariants=None, variables=None,
                      gradients=None,
                      max_iterations=1,
                      integrator='rk2',
                      generator=None,
                      degree=None,
                      itol=1e-12,
                      mode='joint',
                      verbose=False,
                      **kwargs):
    """
    Solve an ODE with homogeneous projection for invariant preservation.
    
    This is a convenience wrapper around scipy.integrate.solve_ivp that automatically
    sets up HomogeneousProjector for invariant preservation. It has the same interface
    as scipy.integrate.solve_ivp but with additional parameters for projection configuration.
    
    Parameters:
    -----------
    fun : callable
        Right-hand side of the system. The calling signature is `fun(t, y)`.
    t_span : 2-tuple of floats
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf.
    y0 : array_like, shape (n,)
        Initial state.
    method : string or OdeSolver class, optional
        Integration method to use (default: 'DOP853'). Can be:
        - String: 'RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
        - OdeSolver class from scipy.integrate._ivp
    invariants : sympy expression or list of sympy expressions, optional
        Invariant(s) to preserve using HomogeneousProjector. If None, no projection 
        is applied and the function behaves like standard solve_ivp.
    variables : list of sympy symbols, optional
        Symbolic variables corresponding to the state vector components.
        Required if using symbolic invariants.
    gradients : callable or list of callables, optional
        Analytical gradient functions for the invariants. If provided, these are used
        instead of numerical finite differences or symbolic differentiation, providing
        significant speedup. Each gradient should accept state as separate arguments:
        grad_i(x1, x2, ..., xn) -> ndarray of shape (n,).
        Must match the number and order of invariants.
    max_iterations : int, optional
        Maximum iterations for the iterative projection (default: 1).
        Setting max_iterations=1 gives a single pseudo-homogeneous projection step.
        Only used for nonlinear projection (when generator is None).
    integrator : str, optional
        Integration method for gradient flow in HomogeneousProjector (default: 'rk2').
        Options: 'euler', 'rk1', 'rk2', 'rk4'.
        Only used when generator is None or when mixing linear/nonlinear projections.
    generator : array_like, str, or list, optional
        Specifies the projection method for each invariant. Can be:
        - None: uses nonlinear HomogeneousProjector for all invariants (default)
        - array: diagonal generator for LinearHomogeneousProjector (single invariant)
        - str: integrator name ('euler', 'rk1', 'rk2', 'rk4') for nonlinear projection
        - list: mix of arrays and strings, one per invariant, enabling mixed 
          linear/nonlinear projections.
    degree : float or list of floats, optional
        Homogeneity degree for each invariant. Required when using linear projection.
        If a single float, same degree is used for all linear projections.
        If a list, must match the number of invariants.
    itol : float, optional
        Absolute tolerance for the invariant error in nonlinear projection (default: 1e-12).
        Projection will iterate until max_iterations is reached or invariant error is below itol.
    mode : str, optional
        Projection mode for multiple invariants when all use nonlinear projection (default: 'joint'):
        - 'joint': Apply all invariants simultaneously in HomogeneousProjector
        - 'alternating': Cycle through invariants, projecting one per step
        Note: When using linear projection or mixed projections, alternating mode is always used.
    verbose : bool, optional
        If True, print information about the projection method used for each invariant (default: False).
    **kwargs
        Additional keyword arguments passed to solve_ivp (rtol, atol, dense_output, etc.).
        
    Returns:
    --------
    Bunch object with the following fields defined:
        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Values of the solution at t.
        sol : OdeSolution or None
            Found solution as OdeSolution instance; None if dense_output was
            set to False.
        t_events : list of ndarray or None
            Contains for each event type a list of arrays at which an event of
            that type event was detected. None if events was None.
        y_events : list of ndarray or None
            For each value of t_events, the corresponding value of the solution.
            None if events was None.
        nfev : int
            Number of evaluations of the right-hand side.
        njev : int
            Number of evaluations of the Jacobian.
        nlu : int
            Number of LU decompositions.
        status : int
            Reason for algorithm termination:
                * -1: Integration step failed.
                *  0: The solver successfully reached the end of tspan.
                *  1: A termination event occurred.
        message : string
            Human-readable description of the termination reason.
        success : bool
            True if the solver reached the interval end or a termination event
            occurred (status >= 0).
      
    """
    if kwargs.get('t_eval', None) is not None:
        print("Warning! When using t_eval with projection, the projected solution is only at specified points.")
        print("For exact invariant preservation at interpolated points, re-evaluate using the dense output followed by projection.")

    # Import method class if it's a string
    if isinstance(method, str):
        solver_class = METHOD_MAP.get(method, DOP853)
    else:
        solver_class = method
    
    # Handle single invariant or multiple invariants
    if not isinstance(invariants, list):
        invariants = [invariants]
    
    # Parse generator specification for each invariant
    if generator is None:
        # All nonlinear
        generators = [None] * len(invariants)
    elif isinstance(generator, list):
        # Mixed or multiple specifications
        generators = generator
    elif isinstance(generator, str):
        # Single string (integrator name) for single invariant
        generators = [generator]
    else:
        # Single array for single invariant (linear)
        generators = [generator]
    
    if len(generators) != len(invariants):
        raise ValueError(f"Number of generators ({len(generators)}) must match number of invariants ({len(invariants)})")
    
    # Parse degree specification
    if degree is None:
        degrees = [None] * len(invariants)
    elif isinstance(degree, list):
        degrees = degree
    else:
        degrees = [degree] * len(invariants)
    
    if len(degrees) != len(invariants):
        raise ValueError(f"Number of degrees ({len(degrees)}) must match number of invariants ({len(invariants)})")
    
    # Parse gradients specification
    if gradients is None:
        grads_list = [None] * len(invariants)
    elif isinstance(gradients, list):
        grads_list = gradients
    else:
        grads_list = [gradients]
    
    if len(grads_list) != len(invariants):
        raise ValueError(f"Number of gradients ({len(grads_list)}) must match number of invariants ({len(invariants)})")
    
    # Check if all generators are nonlinear (None or string integrator names)
    all_nonlinear = all(gen is None or isinstance(gen, str) for gen in generators)
    
    # Validate degrees for nonlinear projections
    for i, (gen, deg) in enumerate(zip(generators, degrees)):
        if (gen is None or isinstance(gen, str)) and deg is not None:
            raise ValueError(f"Invariant {i}: degree should be None when using nonlinear projection")
        if gen is not None and not isinstance(gen, str) and deg is None:
            raise ValueError(f"Invariant {i}: degree must be provided when using linear projection (generator is array)")
    
    if verbose:
        print("=" * 60)
        print("Invariant Preservation Configuration")
        print("=" * 60)
    
    # CASE 1: Joint mode with all nonlinear invariants - create single HomogeneousProjector
    if all_nonlinear and mode == 'joint':
        # Determine which integrator to use (check if all are the same or use default)
        integrator_names = [gen if isinstance(gen, str) else integrator for gen in generators]
        if len(set(integrator_names)) > 1:
            raise ValueError(f"Joint mode requires all invariants to use the same integrator, got: {integrator_names}")
        joint_integrator = integrator_names[0]
        
        if verbose:
            print(f"\nMode: Joint projection for {len(invariants)} invariant(s)")
            print("  Type: Nonlinear homogeneous projection")
            print(f"  Integrator: {joint_integrator}")
            print(f"  Max iterations: {max_iterations}")
            for i, grad in enumerate(grads_list):
                grad_type = "Analytical" if grad is not None else "Numerical (finite differences)"
                print(f"  Invariant {i+1} gradient: {grad_type}")
            print("=" * 60)
        
        # Create single HomogeneousProjector with all invariants
        projector = HomogeneousProjector(
            invariants=invariants,  # Pass list of all invariants
            variables=variables,
            initial_state=y0,
            gradients=grads_list if any(g is not None for g in grads_list) else None,
            integrator=joint_integrator,
            max_iterations=max_iterations,
            tolerance=itol,
        )
    
    # CASE 2: Alternating mode or mixed linear/nonlinear - create individual projectors
    else:
        projectors = []
        
        if verbose:
            mode_str = "alternating" if mode == "alternating" else "alternating (automatic for linear/mixed)"
            print(f"\nMode: {mode_str.capitalize()} projection for {len(invariants)} invariant(s)")
        
        for i, (inv, gen, deg, grad) in enumerate(zip(invariants, generators, degrees, grads_list)):
            if isinstance(gen, str):
                # String generator means nonlinear projection with specified integrator
                if verbose:
                    print(f"\nInvariant {i+1}:")
                    print("  Type: Nonlinear homogeneous projection")
                    print(f"  Integrator: {gen}")
                    print(f"  Max iterations: {max_iterations}")
                    print(f"  Gradient: {'Analytical' if grad is not None else 'Numerical (finite differences)'}")
                
                proj = HomogeneousProjector(
                    invariants=inv,
                    variables=variables,
                    initial_state=y0,
                    gradients=grad,
                    integrator=gen,
                    max_iterations=max_iterations,
                    tolerance=itol,
                )
                projectors.append(proj)
            elif gen is None:
                # None generator means nonlinear projection with default integrator
                if verbose:
                    print(f"\nInvariant {i+1}:")
                    print("  Type: Nonlinear homogeneous projection")
                    print(f"  Integrator: {integrator} (default)")
                    print(f"  Max iterations: {max_iterations}")
                    print(f"  Gradient: {'Analytical' if grad is not None else 'Numerical (finite differences)'}")
                
                proj = HomogeneousProjector(
                    invariants=inv,
                    variables=variables,
                    initial_state=y0,
                    gradients=grad,
                    integrator=integrator,
                    max_iterations=max_iterations,
                    tolerance=itol,
                )
                projectors.append(proj)
            else:
                # Array generator means linear projection
                if verbose:
                    print(f"\nInvariant {i+1}:")
                    print("  Type: Linear homogeneous projection")
                    print(f"  Degree: {deg}")
                    print(f"  Generator diagonal: {gen if isinstance(gen, (list, tuple)) else 'array'}")
                
                proj = LinearHomogeneousProjector(
                    invariant=inv,
                    variables=variables,
                    generator=gen,
                    degree=deg,
                    initial_state=y0
                )
                projectors.append(proj)
        
        if verbose:
            print("=" * 60)
        
        # Create alternating projector
        projector = AlternatingLinearHomogeneousProjector(projectors)
    

    projected_method = create_projected_solver(solver_class, projector)
    
    return scipy.integrate.solve_ivp(fun, t_span, y0, method=projected_method, **kwargs)