"""
Linear Homogeneous Projector for single invariant preservation.
"""

import numpy as np
import sympy as sp
import scipy.linalg


class LinearHomogeneousProjector:
    """Linear Homogeneous Projector - preserves a single homogeneous invariant."""
    
    def __init__(self, invariant, generator, degree, initial_state, phi=None, phi_inv=None, **kwargs):
        self.H_fun = invariant
        if len(generator.shape) == 1:
            self.generator_diag = generator
        elif len(generator.shape) == 2:
            if not np.allclose(generator, np.diag(np.diagonal(generator))):
                raise ValueError("Generator matrix must be diagonal for LinearHomogeneousProjector.")
            self.generator_diag = np.diagonal(generator)
        self.degree = degree
        self.H_initial = self.H_fun(*initial_state)        
        self.phi = phi
        self.phi_inv = phi_inv

    def project(self, y):
        """Apply invariant correction to a state."""
        H_current = self.H_fun(*y)
        if abs(H_current) < 1e-15:
            print(f"Warning: Invariant value = {H_current} or degree = {self.degree} is too close to zero; skipping projection.")
            return y
        y_init = y
        if self.phi is not None:
            y = self.phi(y)
        scale = np.pow((self.H_initial / H_current), (self.generator_diag/self.degree))
        y = scale * y
        if self.phi_inv is not None:
            try:
                y = self.phi_inv(y, x_prev=y_init)
            except TypeError:
                y = self.phi_inv(y)
        return y

class AlternatingLinearHomogeneousProjector:
    """Alternating Linear Homogeneous Projector - preserves multiple homogeneous invariants."""
    def __init__(self, projectors):
        """
        Initialize with a list of LinearHomogeneousProjector instances.
        
        Parameters:
        -----------
        projectors : list
            List of LinearHomogeneousProjector instances
        """
        self.projectors = projectors
        self.current_idx = 0
        self.num_projectors = len(projectors)
        
    def project(self, y):
        """Apply invariant correction to a state."""
        # Apply the current projector
        y_corrected = self.projectors[self.current_idx].project(y)
        
        # Update index to use the next projector next time
        self.current_idx = (self.current_idx + 1) % self.num_projectors
        
        return y_corrected
    