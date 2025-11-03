import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time 
from scipy.integrate import solve_ivp
import sympy as sp

from scipy.integrate._ivp.base import OdeSolver
from homproj.numeric import solve_ivp_fixed_step


def double_pendulum():
    """
    Double pendulum system using Hamiltonian formulation.
    
    Returns:
    --------
    f : function
        Dynamics function f(y) that computes the time derivative
    invariants : list
        List containing the Hamiltonian (energy)
    variables : list
        List of symbolic variables [q1, q2, p1, p2]
    """
    q1, q2, p1, p2 = sp.symbols('q1 q2 p1 p2', real=True)
    variables = [q1, q2, p1, p2]
    
    # Hamiltonian for double pendulum
    H = (p1**2 + 2*p2**2 - 2*sp.cos(q1 - q2)*p1*p2)/(4 - 2*sp.cos(q1 - q2)**2) - 2*sp.cos(q1) - sp.cos(q2)
    
    # Compute Hamilton's equations symbolically
    dq1_dt = sp.diff(H, p1)
    dq2_dt = sp.diff(H, p2)
    dp1_dt = -sp.diff(H, q1)
    dp2_dt = -sp.diff(H, q2)
    
    # Create numerical functions
    dq1_dt_func = sp.lambdify(variables, dq1_dt, 'numpy')
    dq2_dt_func = sp.lambdify(variables, dq2_dt, 'numpy')
    dp1_dt_func = sp.lambdify(variables, dp1_dt, 'numpy')
    dp2_dt_func = sp.lambdify(variables, dp2_dt, 'numpy')
    
    def f(y):
        """Dynamics function for double pendulum"""
        q1_val, q2_val, p1_val, p2_val = y[0], y[1], y[2], y[3]
        dq1 = dq1_dt_func(q1_val, q2_val, p1_val, p2_val)
        dq2 = dq2_dt_func(q1_val, q2_val, p1_val, p2_val)
        dp1 = dp1_dt_func(q1_val, q2_val, p1_val, p2_val)
        dp2 = dp2_dt_func(q1_val, q2_val, p1_val, p2_val)
        return np.array([dq1, dq2, dp1, dp2])
    
    invariants = [H]
    
    return f, invariants, variables


def run_experiment(solver_methods, y0, tmax, tol, timestep, f):
    # Timing and solution storage
    solutions = {}
    timings = {}

    tol_ref = 1e-15
    t_span = (0, tmax) 
    # Run the integrations
    for label, method in solver_methods:
        print(f"Running {label}...")
        start_time = time.time()
        tol = tol_ref if 'reference' in label.lower() else tol
        if isinstance(method, type) and issubclass(method, OdeSolver):
            sol = solve_ivp(f, t_span, y0, method=method, rtol=tol, atol=tol)
        else: # Assume fixed step method for solve_ivp
            sol = solve_ivp_fixed_step(f, t_span, y0, h=timestep, method=method)
        timings[label] = time.time() - start_time
        solutions[label] = sol
        
        if not sol.success:
            print(f"  ❌ Failed: {sol.message}")
            timings[label] = float('inf')
    return solutions, timings

def pendulum_positions(q1, q2, l1=1.0, l2=1.0):
    """Calculate the positions of the pendulum bobs"""
    x1 = l1 * np.sin(q1)
    y1 = -l1 * np.cos(q1)
    x2 = x1 + l2 * np.sin(q2)
    y2 = y1 - l2 * np.cos(q2)
    return x1, y1, x2, y2


def _setup_visualization(fontsize=14):
    """Set up visualization styles for consistency"""
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize - 2,
            "figure.titlesize": fontsize + 2,
            "lines.linewidth": 2,
            "axes.grid": True,
            "grid.alpha": 0.6,
        }
    )


def _determine_time_window(
    solutions, start_time=None, end_time=None, time_threshold=None
):
    """Helper function to determine the time window to display"""
    if end_time is None:
        # Find the maximum end time from all solutions
        end_time = max([sol.t[-1] for name, sol in solutions.items() if sol.success])

    if start_time is None:
        if time_threshold is not None:
            # Use the time threshold from the end time
            start_time = end_time - time_threshold
        else:
            # Default to showing the last 5 seconds if neither start_time nor time_threshold is provided
            start_time = end_time - 5

    return start_time, end_time


def plot_trajectories(solutions, start_time=None, end_time=None, time_threshold=None):
    """
    Create a visualization with a row of trajectory plots for the second pendulum bob.
    Each plot compares one method against the reference solution.

    Parameters:
    -----------
    solutions : dict
        Dictionary mapping method names to their solution objects from solve_ivp
    start_time : float, optional
        Start time of the interval to show (default: None)
    end_time : float, optional
        End time of the interval to show (default: None)
    time_threshold : float, optional
        How many seconds of the end of the simulation to show (default: None)
        If provided and start_time/end_time are None, uses [end_time - time_threshold, end_time]

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the row of trajectory plots
    """
    # Set up visualization style
    _setup_visualization(fontsize=24)

    # Determine the time window
    start_time, end_time = _determine_time_window(
        solutions, start_time, end_time, time_threshold
    )

    # Colors for different methods
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(solutions.keys())
    }
    colors["Reference"] = "black"

    # Find reference solution if it exists
    reference_sol = None
    reference_name = None
    for name, sol in solutions.items():
        if "reference" in name.lower() and sol.success:
            reference_sol = sol
            reference_name = name
            break

    # Get methods to plot (excluding reference)
    methods_to_plot = [
        name
        for name in solutions.keys()
        if "reference" not in name.lower() and solutions[name].success
    ]

    # Create figure with row of trajectory plots
    n_methods = len(methods_to_plot)
    fig = plt.figure(figsize=(4 * n_methods, 4))

    # Create and plot each trajectory
    for i, name in enumerate(methods_to_plot):
        ax = fig.add_subplot(1, n_methods, i + 1)
        sol = solutions[name]

        # Get pendulum positions for this method
        _, _, x2, y2 = pendulum_positions(sol.y[0], sol.y[1])

        # Filter for points in the specified time window
        window_indices = np.where((sol.t >= start_time) & (sol.t <= end_time))[0]

        # Skip if no points are in the time window
        if len(window_indices) == 0:
            continue

        # Extract the window data
        t_last = sol.t[window_indices]
        x2_last = x2[window_indices]
        y2_last = y2[window_indices]

        # Use cubic interpolation for smoother curves
        if len(t_last) > 3:  # Need at least 4 points for cubic interpolation
            # Create parameter along the curve (cumulative distance)
            dx = np.diff(x2_last)
            dy = np.diff(y2_last)
            distances = np.sqrt(dx**2 + dy**2)
            cumulative_dist = np.concatenate(([0], np.cumsum(distances)))

            # Create a fine sampling for smooth curve
            sample_pts = np.linspace(
                cumulative_dist[0], cumulative_dist[-1], num=max(100, 5 * len(t_last))
            )

            # Create interpolation functions
            f_x = interp1d(cumulative_dist, x2_last, kind="cubic")
            f_y = interp1d(cumulative_dist, y2_last, kind="cubic")

            # Generate smooth coordinates
            x_smooth = f_x(sample_pts)
            y_smooth = f_y(sample_pts)

            # Plot the smooth curve
            ax.plot(
                x_smooth, y_smooth, "-", color=colors[name], linewidth=1.5, label=name
            )
        else:
            # Not enough points for interpolation, plot line segments
            ax.plot(
                x2_last, y2_last, "-", color=colors[name], linewidth=1.5, label=name
            )

        # Plot reference solution if available
        if reference_sol is not None:
            _, _, ref_x2, ref_y2 = pendulum_positions(
                reference_sol.y[0], reference_sol.y[1]
            )

            ref_window_indices = np.where(
                (reference_sol.t >= start_time) & (reference_sol.t <= end_time)
            )[0]

            if len(ref_window_indices) > 0:
                ref_x2_last = ref_x2[ref_window_indices]
                ref_y2_last = ref_y2[ref_window_indices]

                if len(ref_window_indices) > 3:
                    # Use cubic interpolation for reference too
                    ref_dx = np.diff(ref_x2_last)
                    ref_dy = np.diff(ref_y2_last)
                    ref_distances = np.sqrt(ref_dx**2 + ref_dy**2)
                    ref_cumulative_dist = np.concatenate(
                        ([0], np.cumsum(ref_distances))
                    )

                    ref_sample_pts = np.linspace(
                        ref_cumulative_dist[0],
                        ref_cumulative_dist[-1],
                        num=max(100, 5 * len(ref_window_indices)),
                    )

                    ref_f_x = interp1d(ref_cumulative_dist, ref_x2_last, kind="cubic")
                    ref_f_y = interp1d(ref_cumulative_dist, ref_y2_last, kind="cubic")

                    ref_x_smooth = ref_f_x(ref_sample_pts)
                    ref_y_smooth = ref_f_y(ref_sample_pts)

                    ax.plot(
                        ref_x_smooth,
                        ref_y_smooth,
                        "-",
                        color=colors[reference_name],
                        linewidth=2.0,
                        label=reference_name,
                        alpha=0.7,
                    )
                else:
                    ax.plot(
                        ref_x2_last,
                        ref_y2_last,
                        "-",
                        color=colors[reference_name],
                        linewidth=2.0,
                        label=reference_name,
                        alpha=0.7,
                    )

        # Set up the position plot
        ax.set_aspect("equal")
        # ax.set_xlim(-2.1, 2.1)
        # ax.set_ylim(-2.1, 2.1)
        ax.set_title(f"{name}")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_analysis_over_initial_conditions(
    solutions_list,
    compute_energy,
    initial_energies,
    start_time=0,
    end_time=None,
    time_threshold=None,
    timings=None,
):
    """
    Create a visualization with mean absolute error vs reference solution, energy error plots,
    and optionally performance comparison. Handles multiple initial conditions.

    Parameters:
    -----------
    solutions_list : list of dict
        List of dictionaries, where each dictionary maps method names to solution objects.
        Each dictionary corresponds to a different initial condition.
        Each should include a reference solution with 'reference' in the name.
    compute_energy : function
        Function to compute energy given a state
    initial_energies : list of float
        List of initial energies corresponding to each set of solutions
    start_time : float, optional
        Start time of the interval to show (default: None)
    end_time : float, optional
        End time of the interval to show (default: None)
    time_threshold : float, optional
        How many seconds of the end of the simulation to show (default: None)
        If provided and start_time/end_time are None, uses [end_time - time_threshold, end_time]
    timings : list of dict, optional
        List of dictionaries mapping method names to execution times.
        If provided, an additional performance comparison plot will be included.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plots
    """
    # Set up visualization style
    _setup_visualization(18)

    # Handle the case where we're given a single solutions dict instead of a list
    if not isinstance(solutions_list, list):
        solutions_list = [solutions_list]
        initial_energies = [initial_energies]
        timings = [timings] if timings is not None else None

    # Determine the time window from the first solutions set
    start_time, end_time = _determine_time_window(
        solutions_list[0], start_time, end_time, time_threshold
    )

    # Colors for different methods - maintain consistent coloring across plots
    method_names = set()
    for solutions in solutions_list:
        method_names.update(solutions.keys())

    # Sort method names to ensure consistent color assignment
    method_names = sorted(list(method_names))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(solutions.keys())
    }
    colors["Reference"] = "black"

    # Create figure with subplots
    n_plots = 3 if timings else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # If we have only one subplot, wrap it in a list
    if n_plots == 1:
        axes = [axes]

    ax_error = axes[0]  # Mean absolute error plot
    ax_energy = axes[1]  # Energy error plot
    if timings:
        ax_perf = axes[2]  # Performance comparison plot

    # Group solutions by method name
    methods_data = {}
    reference_sols = []

    for i, solutions in enumerate(solutions_list):
        # Find reference solution if it exists
        reference_sol = None
        reference_name = None
        for name, sol in solutions.items():
            if "reference" in name.lower() and sol.success:
                reference_sol = sol
                reference_name = name
                break

        if reference_sol is not None:
            reference_sols.append((reference_sol, reference_name))

            # For each method, compute error data
            for name, sol in solutions.items():
                if sol.success and name != reference_name:
                    if name not in methods_data:
                        methods_data[name] = {"error_data": [], "energy_data": []}

                    # Compute mean absolute error between this solution and reference
                    # Create interpolation functions for all state variables
                    ref_interp = []
                    for j in range(4):  # 4 state variables [q1, q2, p1, p2]
                        ref_interp.append(
                            interp1d(
                                reference_sol.t,
                                reference_sol.y[j],
                                kind="cubic",
                                bounds_error=False,
                                fill_value="extrapolate",
                            )
                        )

                    # Interpolate the reference solution to this solution's time points
                    error_sum = np.zeros_like(sol.t)
                    for j in range(4):
                        interp_ref = ref_interp[j](sol.t)
                        error_sum += np.abs(sol.y[j] - interp_ref)

                    # Calculate the mean absolute error across all state variables
                    mean_abs_error = error_sum / 4.0

                    # Store the error data for this method and initial condition
                    methods_data[name]["error_data"].append((sol.t, mean_abs_error))

    # Calculate and store energy errors for each method and initial condition
    for i, (solutions, initial_energy) in enumerate(
        zip(solutions_list, initial_energies)
    ):
        reference_name = reference_sols[i][1] if i < len(reference_sols) else None

        for name, sol in solutions.items():
            if (
                sol.success and name != reference_name
            ):  # Skip reference solution for energy error plot
                # Calculate energy at each time point
                y_data = sol.y.T
                energy = compute_energy(y_data)
                energy_error = np.abs(energy - initial_energy)

                # Store the energy error data
                if name in methods_data:
                    methods_data[name]["energy_data"].append((sol.t, energy_error))

    # Plot mean absolute error with shaded std dev regions
    for name, data in methods_data.items():
        if data["error_data"]:
            # For each time series, interpolate to a common time grid
            common_t = np.linspace(start_time, end_time, 1000)
            interpolated_errors = []

            for t, error in data["error_data"]:
                # Filter time points within our window
                mask = (t >= start_time) & (t <= end_time)
                if np.any(mask):
                    t_filtered = t[mask]
                    error_filtered = error[mask]

                    # Create interpolation function and interpolate to common grid
                    if len(t_filtered) > 1:
                        error_interp = interp1d(
                            t_filtered,
                            error_filtered,
                            kind="cubic",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        interpolated_errors.append(error_interp(common_t))

            # Calculate mean and std dev across all initial conditions
            if interpolated_errors:
                interpolated_errors = np.array(interpolated_errors)
                mean_error = np.mean(interpolated_errors, axis=0)
                std_error = np.std(interpolated_errors, axis=0)

                # Plot mean line
                ax_error.semilogy(
                    common_t,
                    mean_error,
                    "-",
                    color=colors[name],
                    label=name,
                    alpha=0.8,
                    linewidth=2,
                )

                # Plot shaded region for std dev
                ax_error.fill_between(common_t, mean_error - std_error/10, mean_error + std_error,
                                    color=colors[name], alpha=0.2)

    # Plot energy error with shaded std dev regions
    for name, data in methods_data.items():
        if data["energy_data"]:
            # For each time series, interpolate to a common time grid
            common_t = np.linspace(start_time, end_time, 1000)
            interpolated_energy_errors = []

            for t, energy_error in data["energy_data"]:
                # Filter time points within our window
                mask = (t >= start_time) & (t <= end_time)
                if np.any(mask):
                    t_filtered = t[mask]
                    error_filtered = energy_error[mask]

                    # Create interpolation function and interpolate to common grid
                    if len(t_filtered) > 1:
                        error_interp = interp1d(
                            t_filtered,
                            error_filtered,
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        interpolated_energy_errors.append(error_interp(common_t))

            # Calculate mean and std dev across all initial conditions
            if interpolated_energy_errors:
                interpolated_energy_errors = np.array(interpolated_energy_errors)
                mean_error = np.mean(interpolated_energy_errors, axis=0)
                std_error = np.std(interpolated_energy_errors, axis=0)

                # Plot mean line
                ax_energy.semilogy(
                    common_t,
                    mean_error,
                    "-",
                    color=colors[name],
                    label=name,
                    alpha=0.8,
                    linewidth=2,
                )

                # # Plot shaded region for std dev
                ax_energy.fill_between(common_t, mean_error - std_error, mean_error + std_error,
                                     color=colors[name], alpha=0.2)

    # Add performance comparison if timings are provided
    if timings:
        # Group timing data by method
        methods_timing = {}

        for method_timings in timings:
            for name, timing in method_timings.items():
                if name not in methods_timing:
                    methods_timing[name] = []
                if np.isfinite(timing):
                    methods_timing[name].append(timing)

        # Calculate mean and std dev of timings
        # Use the same method order as in solutions to maintain color consistency
        methods = [
            name for name in methods_data.keys() if "reference" not in name.lower()
        ]
        means = []
        stds = []

        for name in methods:
            if name in methods_timing and methods_timing[name]:
                means.append(np.mean(methods_timing[name]))
                stds.append(np.std(methods_timing[name]))
            else:
                means.append(0)
                stds.append(0)

        # Plot the bars with matching colors and error bars
        ax_perf.bar(
            range(len(methods)),
            means,
            yerr=stds,
            color=[colors.get(name) for name in methods],
            capsize=5,
        )

        # Set up the performance plot
        ax_perf.set_title("Wall Clock Time")
        ax_perf.set_ylabel("Time (seconds)")
        ax_perf.set_xticks(range(len(methods)))
        ax_perf.set_xticklabels(methods, rotation=30, ha="right")
        ax_perf.grid(True, alpha=0.3)

    # Set up the error plot
    ax_error.set_title("Mean Absolute Error")
    ax_error.set_xlabel("Time")
    ax_error.grid(True, alpha=0.3)

    # Set up the energy error plot
    ax_energy.set_xlabel("Time")
    ax_energy.set_title("Energy error")
    ax_energy.grid(True, alpha=0.3)

    # Add legends - place all the way to the right outside plots
    if timings:
        # For 3-panel plot, place legend to the right of all plots
        fig.tight_layout(
            rect=[0, 0.05, 0.85, 1]
        )  # Make space for the legend and rotated labels
        handles, labels = ax_energy.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.84, 0.65))
    else:
        # For 2-panel plot
        ax_energy.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    return fig


def create_pendulum_animation(
    solutions,
    compute_energy,
    initial_energy,
    filename="pendulum_animation.gif",
    fps=15,
    time_step=0.1,
):
    """
    Create an animated GIF of the double pendulum simulation with multiple methods.

    Parameters:
    -----------
    solutions : dict
        Dictionary mapping method names to their solution objects from solve_ivp
    compute_energy : function
        Function to compute energy given a state
    initial_energy : float
        The initial energy of the system
    filename : str, optional
        The filename to save the GIF animation (default: 'pendulum_animation.gif')
    fps : int, optional
        Frames per second for the animation (default: 15)
    time_step : float, optional
        Time step between animation frames in simulation time units (default: 0.1)

    Returns:
    --------
    animation : matplotlib.animation.Animation
        The animation object
    """
    from matplotlib.animation import FuncAnimation
    import matplotlib.gridspec as gridspec

    # Visualization setup
    FONTSIZE = 12
    plt.rcParams.update(
        {
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE - 2,
            "figure.titlesize": FONTSIZE + 2,
            "lines.linewidth": 2,
            "axes.grid": True,
            "grid.alpha": 0.6,
        }
    )

    # Colors for different methods
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {
        name: color_cycle[i % len(color_cycle)]
        for i, name in enumerate(solutions.keys())
    }
    colors["Reference"] = "black"

    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1.2])
    ax_pos = fig.add_subplot(gs[0])  # Position plot
    ax_time = fig.add_subplot(gs[1])  # Time series plot
    ax_energy = fig.add_subplot(gs[2])  # Energy error plot

    # Find the common time range for animation (use min of max times)
    max_time = min([sol.t[-1] for name, sol in solutions.items() if sol.success])

    # Create time points for animation
    anim_times = np.arange(0, max_time, time_step)

    # Prepare the figure

    # Set up the position plot
    ax_pos.set_aspect("equal")
    ax_pos.set_xlim(-2.1, 2.1)
    ax_pos.set_ylim(-2.1, 2.1)
    ax_pos.set_title("Double Pendulum Position")
    ax_pos.set_xlabel("x position")
    ax_pos.set_ylabel("y position")
    ax_pos.grid(True, alpha=0.3)

    # Set up the time plot
    ax_time.set_title("X-Position vs Time")
    ax_time.set_xlabel("Time (seconds)")
    ax_time.set_ylabel("X-Position")
    ax_time.set_xlim(0, max_time)

    # Determine y-limits for time series
    all_x2 = []
    for name, sol in solutions.items():
        if sol.success:
            x1, y1, x2, y2 = pendulum_positions(sol.y[0], sol.y[1])
            all_x2.extend(x2)
    if all_x2:
        y_min = min(all_x2) - 0.1
        y_max = max(all_x2) + 0.1
        ax_time.set_ylim(y_min, y_max)

    ax_time.grid(True, alpha=0.3)

    # Set up the energy error plot
    ax_energy.set_xlabel("Time")
    ax_energy.set_ylabel("Energy Error (absolute)")
    ax_energy.set_title("Energy Conservation")
    ax_energy.set_xlim(0, max_time)
    ax_energy.set_yscale("log")

    # Find maximum energy error for setting y-limits
    max_error = 1e-1  # Default
    min_error = 1e-14  # Default
    for name, sol in solutions.items():
        if sol.success:
            y_data = sol.y.T
            energy = compute_energy(y_data)
            energy_error = np.abs(energy - initial_energy)
            if len(energy_error) > 0:
                max_error = max(max_error, np.max(energy_error))
                min_error = min(min_error, np.min(energy_error[energy_error > 0]))

    ax_energy.set_ylim(min_error * 0.5, max_error * 2)
    ax_energy.grid(True, alpha=0.3)

    # Add a time indicator text
    time_text = fig.text(0.5, 0.95, "", ha="center", fontsize=14)

    # Initialize the pendulum artists for each solution
    pendulum_artists = {}
    line_artists = {}
    energy_artists = {}
    history_artists = {}
    history_length = int(2 / time_step)  # Keep 2 seconds of history

    for name, sol in solutions.items():
        if sol.success:
            # Initialize pendulum with lines and markers
            (line1,) = ax_pos.plot([], [], "k-", alpha=0.5, lw=2)  # First pendulum arm
            (line2,) = ax_pos.plot([], [], "k-", alpha=0.5, lw=2)  # Second pendulum arm
            (point1,) = ax_pos.plot(
                [], [], "o", color=colors[name], markersize=10, alpha=0.7
            )  # First pendulum bob
            (point2,) = ax_pos.plot(
                [], [], "o", color=colors[name], markersize=10, alpha=0.7, label=name
            )  # Second pendulum bob

            pendulum_artists[name] = (line1, line2, point1, point2)

            # Time series line
            (line_t,) = ax_time.plot([], [], "-", color=colors[name], lw=2, label=name)
            line_artists[name] = line_t

            # Energy error line
            (line_e,) = ax_energy.plot(
                [], [], "-", color=colors[name], lw=2, label=name
            )
            energy_artists[name] = line_e

            # Position history (trail)
            (history,) = ax_pos.plot([], [], "-", color=colors[name], alpha=0.3, lw=1)
            history_artists[name] = history

    # Add a fixed point at origin for pendulum pivot
    ax_pos.plot(0, 0, "ko", markersize=6)

    # Add legends
    ax_pos.legend(loc="upper right")
    ax_time.legend(loc="upper right")
    ax_energy.legend(loc="upper right")

    # Precompute interpolated values for smoother animation
    interp_data = {}

    for name, sol in solutions.items():
        if sol.success:
            # Interpolate the solution at animation time points
            # Handle non-monotonic time points if any
            t_unique, idx_unique = np.unique(sol.t, return_index=True)
            y_unique = sol.y[:, idx_unique]

            # First, compute energy errors at original solution points
            # This avoids interpolation errors in energy calculation
            original_energy = np.array(
                [compute_energy(y_unique[:, i]) for i in range(y_unique.shape[1])]
            )
            original_energy_error = np.abs(original_energy - initial_energy)

            if len(t_unique) > 1:  # Need at least 2 points for interpolation
                # Interpolate state variables
                y_interp = np.zeros((4, len(anim_times)))
                for i in range(4):  # For each state variable
                    interp_func = interp1d(
                        t_unique,
                        y_unique[i],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                    y_interp[i] = interp_func(anim_times)

                # Compute positions for all animation time points
                x1, y1, x2, y2 = pendulum_positions(y_interp[0], y_interp[1])

                # Interpolate the pre-computed energy errors
                # This preserves the actual energy conservation properties of each method
                energy_error_interp = interp1d(
                    t_unique,
                    original_energy_error,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(anim_times)

                # Store all interpolated data
                interp_data[name] = {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "energy_error": energy_error_interp,
                }

    # Define the animation update function
    def update(frame):
        t = anim_times[frame]
        time_text.set_text(f"Time: {t:.2f}s")

        artists = []

        for name in solutions.keys():
            if name in interp_data:
                data = interp_data[name]

                # Update pendulum position
                line1, line2, point1, point2 = pendulum_artists[name]
                x1, y1 = data["x1"][frame], data["y1"][frame]
                x2, y2 = data["x2"][frame], data["y2"][frame]

                line1.set_data([0, x1], [0, y1])
                line2.set_data([x1, x2], [y1, y2])
                point1.set_data([x1], [y1])
                point2.set_data([x2], [y2])

                # Update time series
                line_t = line_artists[name]
                visible_times = anim_times[: frame + 1]
                visible_x2 = data["x2"][: frame + 1]
                line_t.set_data(visible_times, visible_x2)

                # Update energy error
                line_e = energy_artists[name]
                visible_errors = data["energy_error"][: frame + 1]
                line_e.set_data(visible_times, visible_errors)

                # Update position history (trail)
                history = history_artists[name]
                start_idx = max(0, frame - history_length)
                history.set_data(
                    data["x2"][start_idx : frame + 1], data["y2"][start_idx : frame + 1]
                )

                artists.extend([line1, line2, point1, point2, line_t, line_e, history])

        artists.append(time_text)
        return artists

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=len(anim_times), interval=1000 / fps, blit=True
    )
    plt.tight_layout()

    # Save the animation if filename is provided
    if filename:
        try:
            anim.save(filename, writer="pillow", fps=fps, dpi=72)
            print(f"Animation saved to {filename}")
        except (IOError, ValueError, RuntimeError) as e:
            print(f"Failed to save animation: {e}")

    return anim
