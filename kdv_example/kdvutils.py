"""
Utilities for KdV equation simulation with both spectral and finite difference methods.
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq


class KdVDiscretization:
    """
    Semidiscrete KdV system with support for both spectral and finite difference methods.
    
    Parameters
    ----------
    N : int
        Number of spatial grid points
    L : float
        Domain length [0, L] with periodic boundary conditions
    method : str
        Spatial discretization method: 'spectral' or 'fd' (finite difference)
    fd_accuracy : int
        Accuracy order for finite differences (2, 4, 6, or 8). Only used if method='fd'.
    """
    
    def __init__(self, N=64, L=40.0, method='spectral', fd_accuracy=6, dealias=True):
        self.N = N
        self.L = L
        self.method = method
        self.fd_accuracy = fd_accuracy
        self.dealias = dealias
        
        # Create spatial grid
        self.x = np.linspace(0, L, N, endpoint=False)
        self.dx = L / N
        
        # Setup for spectral method
        if method == 'spectral':
            self.k = 2 * np.pi * fftfreq(N, self.dx)
    
    def spatial_derivative(self, u, order=1):
        """
        Compute spatial derivative of order 1, 2, or 3.
        
        Parameters
        ----------
        u : ndarray
            Function values on grid
        order : int
            Derivative order (1, 2, or 3)
            
        Returns
        -------
        ndarray
            Derivative of u
        """
        if self.method == 'spectral':
            return self._spectral_derivative(u, order)
        elif self.method == 'fd':
            return self._finite_diff_derivative(u, order, self.fd_accuracy)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _spectral_derivative(self, u, order=1):
        """Compute spatial derivative using FFT."""
        u_hat = fft(u)
        if order == 1:
            du_hat = 1j * self.k * u_hat
        elif order == 2:
            du_hat = (1j * self.k)**2 * u_hat
        elif order == 3:
            du_hat = (1j * self.k)**3 * u_hat
        else:
            raise ValueError("Only orders 1, 2 or 3 supported")
        return np.real(ifft(du_hat))
    
    def _finite_diff_derivative(self, u, order=1, accuracy=6):
        """
        Compute spatial derivative using high-order centered finite differences
        with periodic boundary conditions.
        """
        n = len(u)
        du = np.zeros_like(u)
        
        if order == 1:
            # First derivative coefficients
            if accuracy == 2:
                stencil = np.array([-1, 0, 1]) / 2
            elif accuracy == 4:
                stencil = np.array([1, -8, 0, 8, -1]) / 12
            elif accuracy == 6:
                stencil = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60
            elif accuracy == 8:
                stencil = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840
            else:
                raise ValueError(f"Accuracy order {accuracy} not supported for 1st derivative")
            
            # Apply stencil with periodic BC
            width = len(stencil) // 2
            for i in range(n):
                for j, coeff in enumerate(stencil):
                    idx = (i + j - width) % n
                    du[i] += coeff * u[idx]
            du /= self.dx
            
        elif order == 2:
            # Second derivative coefficients
            if accuracy == 2:
                stencil = np.array([1, -2, 1])
            elif accuracy == 4:
                stencil = np.array([-1, 16, -30, 16, -1]) / 12
            elif accuracy == 6:
                stencil = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
            elif accuracy == 8:
                stencil = np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]) / 5040
            else:
                raise ValueError(f"Accuracy order {accuracy} not supported for 2nd derivative")
            
            # Apply stencil with periodic BC
            width = len(stencil) // 2
            for i in range(n):
                for j, coeff in enumerate(stencil):
                    idx = (i + j - width) % n
                    du[i] += coeff * u[idx]
            du /= self.dx**2
            
        elif order == 3:
            # Third derivative: d³/dx³ = d/dx(d²/dx²)
            u_xx = self._finite_diff_derivative(u, order=2, accuracy=accuracy)
            du = self._finite_diff_derivative(u_xx, order=1, accuracy=accuracy)
            
        else:
            raise ValueError("Only orders 1, 2, or 3 supported")
        
        return du
    
    def dealiased_square(self, u):
        N = u.size
        Npad = 3*N//2
        p = (Npad - N)//2
        uh = np.fft.fft(u)
        uhp = np.fft.ifftshift(np.concatenate([np.zeros(p, complex),
                                            np.fft.fftshift(uh),
                                            np.zeros(p, complex)]))
        up  = np.fft.ifft(uhp)              # 1/Npad scaling here
        wp  = up * up
        whp = np.fft.fft(wp)                # cancels one 1/Npad factor
        wh  = np.fft.ifftshift(np.fft.fftshift(whp)[p:-p])
        wh *= Npad / N                      # <<< critical normalization
        return np.fft.ifft(wh).real

    
    def kdv_rhs(self, t, u):
        """
        Right-hand side of semidiscrete KdV equation: u_t = -6uu_x - u_xxx
        
        Note: This is computed as -d/dx(3u^2 + u_xx) to preserve the Hamiltonian structure.
        
        Parameters
        ----------
        t : float
            Current time (unused, but required for ODE solvers)
        u : ndarray
            Current state
            
        Returns
        -------
        ndarray
            Time derivative du/dt
        """
        u_xx = self.spatial_derivative(u, order=2)
        if self.dealias and self.method == 'spectral':
            u_squared = self.dealiased_square(u)
        else:
            u_squared = u * u
        nonlinear_dispersive = 3 * u_squared + u_xx
        return -self.spatial_derivative(nonlinear_dispersive, order=1)
    
    def mass_invariant(self, *u_components):
        """I₁: Mass = ∫ u dx"""
        u = np.array(u_components)
        return self.dx * np.sum(u)
    
    def momentum_invariant(self, *u_components):
        """I₂: Momentum = ∫ u² dx"""
        u = np.array(u_components)
        return self.dx * np.sum(u**2)
    
    def energy_invariant(self, *u_components):
        """I₃: Energy = ∫ (u³ - ½u_x²) dx"""
        u = np.array(u_components)
        u_x = self.spatial_derivative(u, order=1)
        u2 = self.dealiased_square(u)
        return self.dx * (np.sum(u * u2) - 0.5 * np.sum(u_x**2))
    
    def get_invariants(self):
        """Return list of invariant functions."""
        return [self.mass_invariant, self.momentum_invariant, self.energy_invariant]
    
    def get_invariant_gradients(self):
        """Return list of invariant gradient functions."""
        return [self.mass_gradient, self.momentum_gradient, self.energy_gradient]
    
    def mass_gradient(self, *u_components):
        """Gradient of mass invariant."""
        u = np.array(u_components)
        return self.dx * np.ones_like(u)
    
    def momentum_gradient(self, *u_components):
        """Gradient of momentum invariant."""
        u = np.array(u_components)
        return 2 * self.dx * u
    
    def energy_gradient(self, *u_components):
        """Gradient of energy invariant."""
        u = np.array(u_components)
        u_xx = self.spatial_derivative(u, order=2)
        return self.dx * (3 * u**2 + u_xx)
    
    def compute_invariants(self, u):
        """
        Compute all three invariants for a given state.
        
        Returns
        -------
        dict
            Dictionary with keys 'mass', 'momentum', 'energy'
        """
        return {
            'mass': self.mass_invariant(*u),
            'momentum': self.momentum_invariant(*u),
            'energy': self.energy_invariant(*u)
        }
    
    def check_conservation(self, u):
        """
        Check how well the RHS preserves each invariant by computing ∇I · f.
        
        Returns
        -------
        dict
            Dictionary with conservation errors for each invariant
        """
        rhs = self.kdv_rhs(0, u)
        return {
            'mass': np.dot(rhs, self.mass_gradient(*u)),
            'momentum': np.dot(rhs, self.momentum_gradient(*u)),
            'energy': np.dot(rhs, self.energy_gradient(*u))
        }


def soliton(x, c, x0=0.0, t=0.0, L=40.0, tol=1e-12, plot=False):
    """
    Exact single soliton solution for KdV equation with periodic boundaries.
    
    Uses method of images to handle periodic wrapping.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    c : float
        Soliton speed (amplitude = c/2)
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
        Soliton profile u(x, t)
    """
    pos = (x0 + c * t) % L
    u = np.zeros_like(x)
    # Number of images needed so that tail < tol
    width = 2 / np.sqrt(c)  # characteristic half-width
    nmax = int(np.ceil(0.5 * np.log(4 / tol) * width / L)) + 1
    
    for m in range(-nmax, nmax + 1):
        xi = x - pos - m * L
        u += 0.5 * c / np.cosh(0.5 * np.sqrt(c) * xi)**2


    # Plot initial condition
    if plot and t == 0.0:
        plt.figure(figsize=(4, 2))
        plt.plot(x, u, 'b-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('u(x, 0)')
        plt.title('Initial Condition: Single Soliton')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    
    return u


def cnoidal_wave(x, t, L=40.0, m=0.98, u3=0.0, x0=0.0):
    """
    Cnoidal wave solution for KdV equation.
    
    Parameters
    ----------
    x : ndarray
        Spatial grid points
    t : float
        Current time
    L : float
        Domain length (wavelength)
    m : float
        Elliptic modulus (0 < m < 1)
    u3 : float
        Minimum value of solution
    x0 : float
        Phase shift
        
    Returns
    -------
    ndarray
        Cnoidal wave profile u(x, t)
    """
    from scipy.special import ellipk, ellipj
    
    K = ellipk(m)
    kappa = 2.0 * K / L
    A = 2.0 * m * kappa**2
    u2 = u3 + A
    u1 = u3 + A / m
    V = u1 + u2 + u3
    z = kappa * (x - V * t - x0)
    sn, cn, dn, ph = ellipj(z, m)
    
    return u3 + A * cn**2
