"""
Kepler Problem Utilities

This module contains all the utilities for running Kepler problem analysis including:
- Initial condition generation
- Exact solution computation
- Adaptive symplectic integrator
- Plotting functions
- Analysis and comparison utilities
"""

from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from dataclasses import dataclass

# Import the main solver
# from invflow.adaptive import solve_invariant_ivp
from tqdm import tqdm


@dataclass
class SymplecticResult:
    """Result object for symplectic integration."""
    t: np.ndarray
    y: np.ndarray  # shape (4, N)
    nfev: int
    nsteps: int
    nreject: int
    success: bool
    message: str


class KeplerSolver:
    """
    Adaptive Reversible Symplectic Integrator for Kepler Problem.
    
    Uses Störmer-Verlet or Yoshida-4 composition with adaptive step size control.
    """
    def __init__(self, mu=1.0,
                 softening=0.0, 
                 verbose=0,
                 order=4,           # 2 for Störmer–Verlet, 4 for Yoshida-4 composition
                 eps=None,          # controller setpoint ε; if None, taken from h0 in integrate
                 gain=1.0,          # integral gain g in the controller
                 alpha=1.5,         # exponent in Q(r)=r^{-alpha} -> G=-alpha*(p·q)/|q|^2
                 rho_min=1e-12, rho_max=1e12):
        """
        Initialize symplectic integrator.
        
        Parameters:
        -----------
        mu : float
            Gravitational parameter (GM)
        softening : float
            Softening parameter for regularizing close approaches
        verbose : int
            Verbosity level (0=quiet, 1=some output, 2=detailed)
        order : int
            Integration order: 2 for Störmer-Verlet, 4 for Yoshida-4
        eps : float or None
            Controller setpoint ε (if None, taken from h0 in integrate)
        gain : float
            Integral gain in the step size controller
        alpha : float
            Exponent in Q(r)=r^{-alpha} for control function G=-alpha*(p·q)/|q|^2
        rho_min, rho_max : float
            Min/max bounds on step density function
        adaptive : bool
            Whether to use adaptive stepping (True) or fixed step size (False)
        """
        self.mu = float(mu)
        self.verbose = int(verbose)
        self.order = int(order)
        if self.order not in (2, 4, 6, 8):
            raise ValueError("order must be 2, 4, 6 or 8")
        self.eps = None if eps is None else float(eps)
        self.gain = float(gain)
        self.alpha = float(alpha)
        self.eps2 = float(softening)**2
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        if self.order == 2:
            # Störmer-Verlet coefficients
            self.c = np.array([0.5, 0.5], dtype=float)
            self.d = np.array([1.0], dtype=float)
        elif self.order == 4:
            # Yoshida-4 coefficients (composition of S2)
            w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
            w0 = 1.0 - 2.0*w1
            self.c = np.array([w1/2.0, (w0 + w1)/2.0, (w0 + w1)/2.0, w1/2.0], dtype=float)
            self.d = np.array([w1, w0, w1], dtype=float)
        elif self.order == 6:
            # ---- Order 6 (p6 s7), symmetric 7-kick scheme ----
            g6 = np.array([
                0.78451361047755726381949763,
                0.23557321335935813368479318,
            -1.17767998417887100694641568,
                1.31518632068391121888424973,
            -1.17767998417887100694641568,
                0.23557321335935813368479318,
                0.78451361047755726381949763
            ], dtype=float)

            self.d = g6
            self.c = np.concatenate(([g6[0]/2], 0.5*(g6[:-1] + g6[1:]), [g6[-1]/2]))
        elif self.order == 8:
            # ---- Order 8 (p8 s15), symmetric 15-kick scheme ----
            g8 = np.array([
                0.74167036435061295344822780,
            -0.40910082580003159399730010,
                0.19075471029623837995387626,
            -0.57386247111608226665638773,
                0.29906418130365592384446354,
                0.33462491824529818378495798,
                0.31529309239676659663205666,
            -0.79688793935291635401978884,
                0.31529309239676659663205666,
                0.33462491824529818378495798,
                0.29906418130365592384446354,
            -0.57386247111608226665638773,
                0.19075471029623837995387626,
            -0.40910082580003159399730010,
                0.74167036435061295344822780
            ], dtype=float)

            self.d = g8
            self.c = np.concatenate(([g8[0]/2], 0.5*(g8[:-1] + g8[1:]), [g8[-1]/2]))


    def H(self, y):
        """Compute Hamiltonian (energy)."""
        q = y[:2]; p = y[2:]
        r = np.sqrt(q.dot(q) + self.eps2)
        return 0.5*p.dot(p) - self.mu / r

    def force(self, q):
        """Compute gravitational force."""
        r2 = q.dot(q) + self.eps2
        inv_r3 = r2**(-1.5)
        return -self.mu * q * inv_r3

    def G(self, y):
        """Compute step size control function."""
        q = y[:2]; p = y[2:]
        r2 = q.dot(q) + self.eps2
        return -self.alpha * (p.dot(q)) / r2

    def phi(self, y, h, return_feval_count=False, **kwargs):
        """One macro-step with fixed h using symmetric splitting of any order."""
        q = y[:2].copy()
        p = y[2:].copy()
        nfev = 0
        
        # General implementation: alternating drift (c) and kick (d) steps
        # Pattern: c[0] drift, d[0] kick, c[1] drift, d[1] kick, ..., c[-1] drift
        for i in range(len(self.c)):
            # Drift step
            q += self.c[i] * h * p
            
            # Kick step (except for the last iteration)
            if i < len(self.d):
                p += self.d[i] * h * self.force(q)
                nfev += 1
        
        y_new = np.array([q[0], q[1], p[0], p[1]], dtype=float)
        if return_feval_count:
            return y_new, nfev
        else:
            return y_new

    def integrate(self, y0, t_span, h0=None, max_steps=10**7, store_every=1):
        """
        Integrate the Kepler problem with adaptive or fixed step size.
        
        Parameters:
        -----------
        y0 : array_like
            Initial state [qx, qy, px, py]
        t_span : tuple
            Time span (t0, tf)
        h0 : float, optional
            Initial/fixed step size. For adaptive=False, this is the fixed step size.
            For adaptive=True, this is just a hint for initial step size.
        max_steps : int
            Maximum number of steps
        store_every : int
            Store solution every N steps
            
        Returns:
        --------
        SymplecticResult
            Integration result
        """
        t0, tf = float(t_span[0]), float(t_span[1])
        if tf <= t0:
            raise ValueError("t_span must have tf > t0")
        

        print("Adaptive step size integration!")
        # Adaptive step size integration
        # choose ε: prefer user-specified eps, otherwise map h0 -> ε≈h0
        eps = self.eps if self.eps is not None else (float(h0) if h0 is not None else (tf - t0)/1000.0)

        y = np.asarray(y0, dtype=float).copy()
        t = t0
        rho = 1.0  # step density at integer index
        T = [t0]
        Y = [y.copy()]
        nfev = 0
        nsteps = 0

        # Initialize tqdm progress bar to show progress from t0 to tf
        pbar = tqdm(total=tf-t0, desc="Integrating", unit="time", leave=True)
        last_t = t0
        
        while t < tf and nsteps < max_steps:
            # Update progress bar with the time increment and step density
            pbar.update(t - last_t)
            pbar.set_postfix({"rho": f"{rho:.3e}"})
            last_t = t
            # half update of step density with current ε_step
            eps_step = eps
            # predict half-step density
            rho_half = rho + 0.5 * eps_step * self.gain * self.G(y)
            rho_half = float(np.clip(rho_half, self.rho_min, self.rho_max))
            # adjust last step to land exactly at tf
            h = eps_step / rho_half
            # if t + h > tf:
            #     eps_step = (tf - t) * rho_half
            #     rho_half = rho + 0.5 * eps_step * self.gain * self.G(y)
            #     rho_half = float(np.clip(rho_half, self.rho_min, self.rho_max))
            #     h = eps_step / rho_half

            y_new, nf = self.phi(y, h, return_feval_count=True); nfev += nf
            # second half update of step density using y_{n+1}
            rho = rho_half + 0.5 * eps_step * self.gain * self.G(y_new)
            rho = float(np.clip(rho, self.rho_min, self.rho_max))

            t += h
            nsteps += 1
            y = y_new
            if (nsteps % store_every) == 0 or t >= tf:
                T.append(t); Y.append(y.copy())
            if self.verbose and (nsteps % 1000) == 0:
                print(f"step {nsteps}, t={t:.6f}, h={h:.3e}, rho_half={rho_half:.3e}")
        
        success = (t >= tf)
        msg = "finished" if success else "stopped (step limit)"
        T = np.array(T, dtype=float)
        Y = np.array(Y, dtype=float).T
    
        return SymplecticResult(T, Y, nfev, nsteps, 0, success, msg)



def _rot90(v): 
    """Rotate vector by 90 degrees."""
    return np.array([-v[1], v[0]])


def _wrap_pm_pi(x): 
    """Wrap angle to [-π, π]."""
    return (x + np.pi) % (2*np.pi) - np.pi


def _solve_E_danby(M, e, tol=1e-14, maxit=12):
    """
    Solve Kepler's equation E - e*sin(E) = M using Danby's method.
    
    Parameters:
    -----------
    M : float
        Mean anomaly
    e : float
        Eccentricity
    tol : float
        Convergence tolerance
    maxit : int
        Maximum iterations
        
    Returns:
    --------
    float
        Eccentric anomaly E
    """
    M = _wrap_pm_pi(M)
    if e < 0.8:
        E = M + e*np.sin(M) + 0.5*e*e*np.sin(2*M)
    else:
        E = M + np.sign(np.sin(M))*0.85*e
    for _ in range(maxit):
        sE, cE = np.sin(E), np.cos(E)
        f   = E - e*sE - M
        fp  = 1.0 - e*cE
        fpp = e*sE
        fppp= e*cE
        d1 = -f/fp
        d2 = -f/(fp + 0.5*d1*fpp)
        d3 = -f/(fp + 0.5*d2*fpp + (1.0/6.0)*d2*d2*fppp)
        En = E + d3
        if abs(En - E) < tol:
            return En
        E = En
    return E


def kepler_exact_elliptic(y0, t, mu=1.0):
    """
    Compute exact solution to Kepler problem for elliptic orbits.
    
    Parameters:
    -----------
    y0 : array_like
        Initial state [qx, qy, px, py]
    t : array_like
        Times at which to evaluate solution
    mu : float
        Gravitational parameter
        
    Returns:
    --------
    ndarray
        Solution at times t, shape (len(t), 4)
    """
    y0 = np.asarray(y0, float)
    q0, p0 = y0[:2], y0[2:]
    r0 = np.linalg.norm(q0); v0 = np.linalg.norm(p0)
    eps = 0.5*v0*v0 - mu/r0
    if eps >= 0:
        raise ValueError("non-elliptic orbit; this solver assumes ε<0")
    a = -mu/(2.0*eps)
    h = q0[0]*p0[1] - q0[1]*p0[0]
    beta = np.dot(q0, p0)
    evec = ((v0*v0 - mu/r0)*q0 - beta*p0)/mu
    e = np.linalg.norm(evec)

    if e < 1e-14:
        u = q0/r0
        v = _rot90(u);  v = v if h >= 0 else -v
        E0 = np.arctan2(np.dot(q0, v)/a, np.dot(q0, u)/a); M0 = E0
    else:
        u = evec/e
        v = _rot90(u);  v = v if h >= 0 else -v
        x0 = np.dot(q0, u); y0p = np.dot(q0, v)
        fac = np.sqrt(max(1.0 - e*e, 0.0))
        cosE0 = x0/a + e
        sinE0 = y0p/(a*fac if fac>0 else np.inf)
        cosE0 = np.clip(cosE0, -1.0, 1.0)
        E0 = np.arctan2(sinE0, cosE0)
        M0 = E0 - e*np.sin(E0)

    n = np.sqrt(mu/(a*a*a))
    B = np.column_stack((u, v))
    t = np.atleast_1d(t).astype(float)
    Y = np.zeros((t.size, 4))
    fac = np.sqrt(max(1.0 - e*e, 0.0))

    for i, ti in enumerate(t):
        M = M0 + n*ti
        E = _solve_E_danby(M, e)
        cE, sE = np.cos(E), np.sin(E)
        x_pf = a*(cE - e)
        y_pf = a*(fac*sE)
        r = a*(1.0 - e*cE)
        vx_pf = -a*n*sE / r
        vy_pf =  a*n*fac*cE / r
        q = B @ np.array([x_pf, y_pf])
        p = B @ np.array([vx_pf, vy_pf])
        Y[i,:2], Y[i,2:] = q, p

    return Y[0].T if Y.shape[0] == 1 else Y.T

def compute_exact_solutions(solutions, y0):
    """
    Compute exact solutions for all methods at their respective time points.
    
    Parameters:
    -----------
    solutions : dict
        Dictionary of solution objects keyed by method name
    y0 : array_like
        Initial conditions
        
    Returns:
    --------
    dict
        Dictionary of exact solutions keyed by method name
    """
    exact_solutions = {}
    for method_name, sol in solutions.items():
        if method_name == 'symplectic':
            time_points = sol.t
        else:
            time_points = sol.t
        
        exact_sol = kepler_exact_elliptic(y0, time_points)
        exact_solutions[method_name] = exact_sol
        
    return exact_solutions

def plot_kepler_trajectories(solutions, y_ref, colors=None, aspect_ratio=0.5):
    """
    Plot Kepler trajectories with adjustable aspect ratio.
    
    Parameters:
    -----------
    solutions : dict
        Dictionary of solution objects keyed by method name
    y_ref : ndarray
        Reference trajectory for comparison
    colors : dict, optional
        Dictionary mapping method names to colors
    aspect_ratio : float, optional
        Controls the y-to-x aspect ratio of the plot (default: 0.5)
        Values less than 1 make the plot wider than tall
    """
    n_methods = len(solutions)
    _init_plots(fontsize=14)
    if colors is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(solutions.keys())}

    # Create a grid with 2 rows
    n_cols = int(np.ceil(n_methods / 2))
    fig1, axes1 = plt.subplots(2, n_cols, figsize=(4*n_cols, 5))
    
    # Make axes1 a 2D array even if there are fewer than 2 methods
    if n_methods == 1:
        axes1 = np.array([[axes1[0]], [axes1[1]]])
    elif n_cols == 1:
        axes1 = axes1.reshape(2, 1)
    
    # Flatten for easier iteration
    axes_flat = axes1.flatten()
    
    for idx, (name, sol) in enumerate(solutions.items()):
        if idx < len(axes_flat):
            ax = axes_flat[idx]
            y_data = sol.y.T
            
            # Plot trajectory with dots
            ax.plot(y_data[:, 0], y_data[:, 1], '.', color=colors[name], 
                    alpha=0.8, markersize=3, label='Trajectory')
            
            ax.plot(y_ref[0, :], y_ref[1, :], '-k', 
                    alpha=0.7, linewidth=2, label='Reference (Exact)')
            
            # Plot central body
            ax.plot(0, 0, 'ko', markersize=8, label='Central Body')
            
            # Apply custom aspect ratio
            ax.set_box_aspect(aspect_ratio)
            
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_title(f'{name}')
            ax.grid(True, alpha=0.3)
            
    # Hide unused subplots
    for idx in range(n_methods, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig1

def _init_plots(fontsize=18):
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize - 2,
        'figure.titlesize': fontsize + 2,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.6,
    })



def plot_kepler_errors(solutions, 
                        compute_energy,
                        compute_angular_momentum,
                        compute_third_invariant,
                        colors=None,
                        n_plot=200):
    _init_plots(fontsize=18)
    if colors is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(solutions.keys())}
    linewidth = 2
    alpha=0.7
    left_ax_limit = 5
    fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))

    def compute_averaged_errors(t, errors, n_points):
        """Compute averaged errors over n_points intervals."""
        if len(errors) <= n_points:
            return t, errors
        
        n_total = len(errors)
        step_size = max(1, n_total // n_points)
        
        t_avg = []
        errors_avg = []
        
        for i in range(0, n_total, step_size):
            end_idx = min(i + step_size, n_total)
            if end_idx > i:
                t_avg.append(np.mean(t[i:end_idx]))
                errors_avg.append(np.mean(errors[i:end_idx]))
        
        return np.array(t_avg), np.array(errors_avg)

    # Get initial invariants from the first solution initial state
    ax = axes2[0]
    solutions_list = list(solutions.values())
    y0 = solutions_list[0].y.T[0]
    initial_energy = compute_energy(y0)
    initial_angular_momentum = compute_angular_momentum(y0)
    initial_third_invariant = compute_third_invariant(y0)

    # Energy Conservation (log-log)
    for name, sol in solutions.items():
        y_data = sol.y.T
        energy = compute_energy(y_data)
        energy_error = np.abs(energy - initial_energy)
        t_avg, error_avg = compute_averaged_errors(sol.t, energy_error, n_plot)
        
        # Exclude the last point for plotting
        t_plot = t_avg[:-1]
        error_plot = error_avg[:-1]
        
        # Filter out zero errors for log scale
        nonzero_mask = error_plot > 1e-16
        if np.any(nonzero_mask):
            ax.loglog(t_plot[nonzero_mask], error_plot[nonzero_mask], '-', 
                    color=colors[name], label=name, markersize=3, linewidth=linewidth, alpha=alpha)
            ax.set_xlim(left=left_ax_limit)

    ax.set_xlabel('Time')
    ax.set_ylabel('Error')
    ax.set_title(f'Energy')
    ax.grid(True, alpha=0.3)

    # Angular Momentum Conservation (log-log)
    ax = axes2[1]
    for name, sol in solutions.items():
        y_data = sol.y.T
        angular_momentum = compute_angular_momentum(y_data)
        momentum_error = np.abs(angular_momentum - initial_angular_momentum)
        t_avg, error_avg = compute_averaged_errors(sol.t, momentum_error, n_plot)
        
        # Exclude the last point for plotting
        t_plot = t_avg[:-1]
        error_plot = error_avg[:-1]
        
        # Filter out zero errors for log scale
        nonzero_mask = error_plot > 1e-16
        if np.any(nonzero_mask):
            ax.loglog(t_plot[nonzero_mask], error_plot[nonzero_mask], '-', 
                    color=colors[name], label=name, markersize=3, linewidth=linewidth, alpha=alpha)
            ax.set_xlim(left=left_ax_limit)

    ax.set_xlabel('Time')
    ax.set_title(f'Angular Momentum')
    ax.grid(True, alpha=0.3)

    # Third Invariant Conservation (log-log)
    ax = axes2[2]
    for name, sol in solutions.items():
        y_data = sol.y.T
        third_invariant = compute_third_invariant(y_data)
        third_error = np.abs(third_invariant - initial_third_invariant)
        t_avg, error_avg = compute_averaged_errors(sol.t, third_error, n_plot)
        
        # Exclude the last point for plotting
        t_plot = t_avg[:-1]
        error_plot = error_avg[:-1]
        
        # Filter out zero errors for log scale
        nonzero_mask = error_plot > 1e-16
        if np.any(nonzero_mask):
            ax.loglog(t_plot[nonzero_mask], error_plot[nonzero_mask], '-', 
                    color=colors[name], label=name, markersize=3, linewidth=linewidth, alpha=alpha)
            ax.set_xlim(left=left_ax_limit)

    ax.set_xlabel('Time')
    ax.set_title(f'Laplace–Runge–Lenz')
    ax.grid(True, alpha=0.3)

    # Total MSE Error vs Exact Solution (log-log)
    ax = axes2[3]
    for name, sol in solutions.items():
        
        # Compute MAE error at each time point
        mae_error = sol.error
        t_avg, error_avg = compute_averaged_errors(sol.t, mae_error, n_plot)
        
        # Exclude the last point for plotting
        t_plot = t_avg[:-1]
        error_plot = error_avg[:-1]
        
        # Filter out zero errors for log scale
        nonzero_mask = error_plot > 1e-16
        if np.any(nonzero_mask):
            ax.loglog(t_plot[nonzero_mask], error_plot[nonzero_mask], '-', 
                        color=colors[name], label=name, markersize=3, linewidth=linewidth, alpha=alpha)
            ax.set_xlim(left=left_ax_limit)

    ax.set_xlabel('Time')
    ax.set_title(f'MAE')
    ax.grid(True, alpha=0.3)
    # Place the legend to the right of the plots, on a separate axis
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)


    plt.tight_layout()
    plt.show()


    return fig2



def plot_cost_metrics(solutions, colors=None):
    _init_plots(fontsize=12)
    if colors is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(solutions.keys())}
    # ============================================================================
    # FIGURE 3: PERFORMANCE METRICS (Function evaluations, runtime, step sizes)
    # ============================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(6, 4))

    # Function Evaluations (bars)
    ax = axes3[0]
    methods = list(solutions.keys())
    nfevs = [solutions[name].nfev for name in methods]
    bars = ax.bar(range(len(methods)), nfevs, color=[colors.get(name, 'gray') for name in methods])
    ax.set_title('Function Evaluations')
    ax.set_ylabel('Number')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([name for name in methods], rotation=45, ha='right')
    ax.set_ylim(0, max(nfevs) * 1.15/10)
    ax.grid(True, alpha=0.5)

    # Wall Clock Time (bars)
    ax = axes3[1]
    methods = list(solutions.keys())
    times = [solutions[name].runtime if np.isfinite(solutions[name].runtime) else 0 for name in methods]
    bars = ax.bar(range(len(methods)), times, color=[colors.get(name, 'gray') for name in methods])
    
    # Add integer labels to the bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(times) * 0.02,
                f'{int(times[i])}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Time (seconds)')
    ax.set_ylim(0, max(times) * 1.15)
    ax.set_xticks(range(len(methods)))
    ax.set_title('Wall Clock Time')
    ax.set_xticklabels([name for name in methods], rotation=45, ha='right')

    plt.tight_layout()
    plt.show()
    return fig3


def plot_error_performance_analysis(df_results, legend=True, plot_fevals=True):
    _init_plots(fontsize=12)
    linewidth = 2
    # Create single plot with dual x-axes
    fig, ax_runtime = plt.subplots(1, 1, figsize=(5, 4))

    if plot_fevals:
        # Create second x-axis for function evaluations
        ax_fevals = ax_runtime.twiny()

    # Filter successful runs only
    df_success = df_results[df_results['success'] & 
                        (df_results['final_error'] != float('inf')) & 
                        (df_results['nfev'] != float('inf'))].copy()

    # Create color scheme for methods
    methods = df_success['Method'].unique()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {method: color_cycle[i % len(color_cycle)] for i, method in enumerate(methods)}

    # Plot both runtime and fevals for each method
    for method in methods:
        method_data = df_success[df_success['Method'] == method]
        if not method_data.empty:
            # Sort by runtime for runtime plot
            method_data_runtime = method_data.sort_values('runtime')
            ax_runtime.loglog(method_data_runtime['runtime'], method_data_runtime['final_error'],
                            'o-', color=colors[method], label=f'{method}', 
                            markersize=4, linewidth=linewidth, alpha=0.8)
            if plot_fevals:             
                # Sort by nfev for fevals plot
                method_data_fevals = method_data.sort_values('nfev')
                ax_fevals.loglog(method_data_fevals['nfev'], method_data_fevals['final_error'],
                            'o--', color=colors[method], 
                            markersize=4, linewidth=linewidth, alpha=0.8)

    # Set labels and formatting
    ax_runtime.set_xlabel('Runtime (seconds)' )
    ax_runtime.set_ylabel('Final Error')
    # ax_runtime.set_xlim(left=0.1)
    if plot_fevals:
        ax_fevals.set_xlabel('Function Evaluations' )
    
    if legend: ax_runtime.legend()
    plt.tight_layout()
    return fig




def plot_step_sizes(solutions, n_periods=2, colors=None, last_only=False):
    """Plot step size as a function of time for all solutions.
    
    If last_only is True, plot the last n_periods*2*pi interval instead of the first.
    """
    _init_plots(fontsize=12)
    period = 2 * np.pi
    if colors is None:
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        line_cycle = ['o', 'd', '*', 's', '^', 'x', '<', '>', 'p', 'v']
        colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(solutions.keys())}
        lines = {name: line_cycle[i % len(line_cycle)] for i, name in enumerate(solutions.keys())}
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))  # Increased figure width to accommodate legend
    
    for method_name, sol in solutions.items():
        color = colors.get(method_name, 'blue')
        line = lines.get(method_name, 'o')

        # Check if solution has step size information
        if hasattr(sol, 't') and len(sol.t) > 1:
            # Calculate step sizes
            dt = np.diff(sol.t)
            t_mid = sol.t[:-1] + dt/2  # Midpoint times for step sizes

            # Select time window
            if last_only:
                t_end = sol.t[-1]
                t_start = t_end - n_periods * period
                mask = (t_mid >= t_start) & (t_mid <= t_end)
                ax.set_xlim((t_start, t_end))
            else:
                t_start = sol.t[0]
                t_end = t_start + n_periods * period
                mask = (t_mid >= t_start) & (t_mid <= t_end)
                ax.set_xlim((t_start, t_end))
            
            # Plot step sizes
            ax.semilogy(t_mid[mask], dt[mask], line, label=method_name, color=color, alpha=0.6)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Step Size')
    ax.set_title('Step Size Evolution')
    ax.grid(True, alpha=0.3)
    
    # Place the legend to the right of the figure
    ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    
    plt.tight_layout()
    return fig
