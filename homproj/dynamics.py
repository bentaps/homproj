import numpy as np
import sympy as sp

class HamiltonianSystem:
    def __init__(self, H_expr, x_syms):
        self.x_syms = list(x_syms)
        self.d = len(self.x_syms)
        assert self.d % 2 == 0
        self.n = self.d // 2

        # Canonical symplectic matrix
        self.J = sp.Matrix.vstack(
            sp.Matrix.hstack(sp.zeros(self.n), sp.eye(self.n)),
            sp.Matrix.hstack(-sp.eye(self.n), sp.zeros(self.n)),
        )

        # Symbolic objects
        self.gradH_expr = sp.Matrix([sp.diff(H_expr, s) for s in self.x_syms])
        self.f_expr = self.J * self.gradH_expr

        # Numpy callables
        self.H_lambda = sp.lambdify(self.x_syms, H_expr, "numpy")
        self.gradH_lambda = sp.lambdify(self.x_syms, self.gradH_expr, "numpy")
        self.f_lambda = sp.lambdify(self.x_syms, self.f_expr, "numpy")

    # Numpy front-ends
    def H(self, x):
        comps = [x[..., i] for i in range(self.d)]
        return self.H_lambda(*comps)

    def gradH(self, x):
        comps = [x[..., i] for i in range(self.d)]
        val = self.gradH_lambda(*comps)
        # Handle the case where val is a nested structure (sympy Matrix lambdified)
        if hasattr(val, "shape") and len(val.shape) > 1:
            # Flatten and take first column if it's a column vector
            val = val.flatten()
        return np.array([float(val[i]) for i in range(self.d)])

    def f(self, x):
        comps = [x[..., i] for i in range(self.d)]
        val = self.f_lambda(*comps)
        # Handle the case where val is a nested structure (sympy Matrix lambdified)
        if hasattr(val, "shape") and len(val.shape) > 1:
            # Flatten and take first column if it's a column vector
            val = val.flatten()
        return np.array([float(val[i]) for i in range(self.d)])

