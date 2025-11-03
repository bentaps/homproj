import numpy as np

class CHDiscretization:
    def __init__(self, N=64, L=40.0):
        self.N = N
        self.L = L
        self.x = np.linspace(0.0, L, N, endpoint=False)
        self.dx = L / N

        # Precompute inverse of (I - D^{(2)}) for solving linear system
        # Build pentadiagonal matrix for 4th-order accurate D^{(2)}
        self._build_helmholtz_inverse()

    def _build_helmholtz_inverse(self):
        """Build and invert (I - D^{(2)}) using 4th-order finite differences."""
        N = self.N
        dx2 = self.dx ** 2
        
        # 4th-order centered D^{(2)}: (-u_{i+2} + 16u_{i+1} - 30u_i + 16u_{i-1} - u_{i-2}) / (12*dx^2)
        # So D^{(2)} coefficients: [-1/12, 4/3, -5/2, 4/3, -1/12] / dx^2
        
        # Build pentadiagonal matrix for I - D^{(2)}
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu
        
        # Diagonals for periodic BC
        main_diag = np.ones(N) + (5.0 / 2.0) / dx2  # 1 - (-5/2)/dx^2
        off1_diag = np.ones(N) * (-4.0 / 3.0) / dx2  # -4/3 / dx^2
        off2_diag = np.ones(N) * (1.0 / 12.0) / dx2  # 1/12 / dx^2
        
        # Create sparse matrix with periodic wrapping
        diagonals = [main_diag, off1_diag, off1_diag, off2_diag, off2_diag]
        offsets = [0, 1, -1, 2, -2]
        A = diags(diagonals, offsets, shape=(N, N), format='lil')
        
        # Wrap around for periodic BC (connect first and last points)
        A[0, -1] = off1_diag[0]
        A[0, -2] = off2_diag[0]
        A[1, -1] = off2_diag[1]
        A[-1, 0] = off1_diag[-1]
        A[-1, 1] = off2_diag[-1]
        A[-2, 0] = off2_diag[-2]
        
        # LU factorization for fast solve
        self._helmholtz_lu = splu(A.tocsc())

    # ---------- basic periodic shift helpers ----------
    def _rollp(self, u, k=1):  # u_{i+k}
        return np.roll(u, -k)

    def _rollm(self, u, k=1):  # u_{i-k}
        return np.roll(u, k)

    # ---------- high-order finite difference operators ----------
    def delta1(self, u):
        """4th-order centered first derivative: D^{(1)} u"""
        # Stencil: [1/12, -2/3, 0, 2/3, -1/12] / dx
        return (self._rollm(u, 2) - 8.0 * self._rollm(u, 1) + 
                8.0 * self._rollp(u, 1) - self._rollp(u, 2)) / (12.0 * self.dx)

    def delta2(self, u):
        """4th-order centered second derivative: D^{(2)} u"""
        # Stencil: [-1/12, 4/3, -5/2, 4/3, -1/12] / dx^2
        return (-self._rollp(u, 2) + 16.0 * self._rollp(u, 1) - 30.0 * u + 
                16.0 * self._rollm(u, 1) - self._rollm(u, 2)) / (12.0 * self.dx ** 2)

    def delta_plus(self, u):
        """Forward difference (for averaging): D^+ u"""
        return (self._rollp(u) - u) / self.dx

    def delta_minus(self, u):
        """Backward difference (for averaging): D^- u"""
        return (u - self._rollm(u)) / self.dx

    def mu_plus(self, u):
        """Forward average: M^+ u"""
        return 0.5 * (self._rollp(u) + u)

    def mu_minus(self, u):
        """Backward average: M^- u"""
        return 0.5 * (u + self._rollm(u))

    # ---------- Hamiltonian gradient and RHS ----------
    def grad_H2(self, u):
        """Gradient of H2 Hamiltonian: δH2/δu"""
        u2 = u * u
        term1 = 1.5 * u2
        term2 = 0.5 * self.mu_minus(self.delta_plus(u) ** 2)
        term3 = -0.5 * self.delta2(u2)
        return term1 + term2 + term3

    def _solve_I_minus_D2(self, f):
        """Solve (I - D^{(2)}) y = f using precomputed LU factorization."""
        return self._helmholtz_lu.solve(f)

    def ch_rhs(self, t, u):
        """Camassa-Holm RHS: du/dt = -(I - D^{(2)})^{-1} D^{(1)} grad_H2"""
        g = self.grad_H2(u)
        w = self.delta1(g)
        y = self._solve_I_minus_D2(w)
        return -y

    # ---------- discrete invariants ----------
    def H1_invariant(self, u):
        ux2 = 0.5 * (self.delta_plus(u) ** 2 + self.delta_minus(u) ** 2)
        return 0.5 * self.dx * np.sum(u ** 2 + ux2)

    def H2_invariant(self, u):
        ux2 = 0.5 * (self.delta_plus(u) ** 2 + self.delta_minus(u) ** 2)
        return 0.5 * self.dx * np.sum(u ** 3 + u * ux2)

    def compute_invariants(self, u):
        return {"H1": self.H1_invariant(u), "H2": self.H2_invariant(u)}

    def check_conservation(self, u):
        rhs = self.ch_rhs(0.0, u)
        grad = self.grad_H2(u)
        # Discrete L2 inner product with weight Δx
        return {"dH2_dt": self.dx * float(np.dot(grad, rhs))}
    
    def get_invariants(self):
        """Return list of invariant functions for projection."""
        return [
            lambda *u_comp: self.H1_invariant(np.array(u_comp)),
            lambda *u_comp: self.H2_invariant(np.array(u_comp))
        ]


def peakon(x, c, x0=0.0, t=0.0, L=40.0, tol=1e-12):
    """
    Exact peakon solution for Camassa-Holm equation with periodic boundaries.
    
    Single peakon: u(x,t) = c * exp(-|x - ct - x0|)
    
    Uses method of images to handle periodic wrapping.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    c : float
        Peakon amplitude and speed
    x0 : float
        Initial position
    t : float
        Current time
    L : float
        Domain length (period)
    tol : float
        Tolerance for tail truncation
        
    Returns
    -------
    ndarray
        Peakon profile u(x, t)
    """
    pos = (x0 + c * t) % L
    u = np.zeros_like(x)
    
    # Characteristic decay length
    decay_length = 1.0  # exp(-x) decays on scale 1
    
    # Number of images needed
    nmax = int(np.ceil(np.log(1 / tol) * decay_length / L)) + 1
    
    for m in range(-nmax, nmax + 1):
        xi = x - pos - m * L
        u += c * np.exp(-np.abs(xi))
    
    return u


def smooth_initial_condition(x, L=40.0, amplitude=1.0):
    """
    Create a smooth initial condition (e.g., Gaussian or cosine wave).
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    L : float
        Domain length
    amplitude : float
        Wave amplitude
        
    Returns
    -------
    ndarray
        Initial condition u(x, 0)
    """
    # Smooth wave packet
    return amplitude * np.exp(-((x - L/2) / (L/10))**2) * np.cos(2 * np.pi * x / L * 2)
