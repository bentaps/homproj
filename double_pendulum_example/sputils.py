import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def spring_pendulum_positions(r, theta):
    """Calculate the positions of the spring pendulum bob"""
    x = r * np.sin(theta)
    y = -r * np.cos(theta)
    return x, y

def _setup_visualization(fontsize=14):
    """Set up visualization styles for consistency"""
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

def _determine_time_window(solutions, start_time=None, end_time=None, time_threshold=None):
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
    Create a visualization with a row of trajectory plots for the spring pendulum.
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
    start_time, end_time = _determine_time_window(solutions, start_time, end_time, time_threshold)
    start_time = np.max([start_time, 0])
    end_time = np.min([end_time, max([sol.t[-1] for name, sol in solutions.items() if sol.success])])
    # Colors for different methods - use consistent color assignment as in plot_analysis_over_initial_conditions
    # First collect all method names
    method_names = sorted(list(solutions.keys()))
    
    # Sort method names to ensure consistent color assignment
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(method_names)}
    colors['Reference'] = 'black'
    
    # Find reference solution if it exists
    reference_sol = None
    reference_name = None
    for name, sol in solutions.items():
        if 'reference' in name.lower() and sol.success:
            reference_sol = sol
            reference_name = name
            break
    
    # Get methods to plot (excluding reference)
    methods_to_plot = [name for name in solutions.keys() 
                      if 'reference' not in name.lower() and solutions[name].success]
    
    # Create figure with row of trajectory plots
    n_methods = len(methods_to_plot)
    fig = plt.figure(figsize=(4*n_methods, 4))
    
    # Create and plot each trajectory
    for i, name in enumerate(methods_to_plot):
        ax = fig.add_subplot(1, n_methods, i+1)
        sol = solutions[name]
        
        # Add method name as title in black
        ax.set_title(f'{name}', color='black')
        
        # Get pendulum positions for this method (theta and r are the first two state variables)
        x, y = spring_pendulum_positions(sol.y[0], sol.y[1])
        
        # Filter for points in the specified time window
        window_indices = np.where((sol.t >= start_time) & (sol.t <= end_time))[0]
        
        # Skip if no points are in the time window
        if len(window_indices) == 0:
            continue
        
        # Extract the window data
        t_last = sol.t[window_indices]
        x_last = x[window_indices]
        y_last = y[window_indices]
        
        mark_sol_points = False

        # Use cubic interpolation for smoother curves
        if len(t_last) > 3:  # Need at least 4 points for cubic interpolation
            # Create parameter along the curve (cumulative distance)
            dx = np.diff(x_last)
            dy = np.diff(y_last)
            distances = np.sqrt(dx**2 + dy**2)
            cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
            
            # Create a fine sampling for smooth curve
            sample_pts = np.linspace(cumulative_dist[0], cumulative_dist[-1], 
                                    num=max(100, 5*len(t_last)))
            
            # Create interpolation functions
            f_x = interp1d(cumulative_dist, x_last, kind='cubic')
            f_y = interp1d(cumulative_dist, y_last, kind='cubic')
            
            # Generate smooth coordinates
            x_smooth = f_x(sample_pts)
            y_smooth = f_y(sample_pts)
            
            # Define colors with increasing opacity based on time progression
            n_segments = 20  # Number of segments with varying opacity
            segment_size = len(x_smooth) // n_segments
            
            # Plot the trajectory with increasing opacity
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(x_smooth)
                
                # Calculate opacity from 0.2 to 1.0 based on segment index
                opacity = 0.2 + 0.8 * i / (n_segments - 1)
                
                ax.plot(x_smooth[start_idx:end_idx], y_smooth[start_idx:end_idx], '-', 
                        color=colors[name], linewidth=3.0, alpha=opacity)
            
            # If requested, mark the actual solution points with markers
            if mark_sol_points:
                # Use time-based coloring for markers to match the segment coloring
                t_norm = (t_last - t_last[0]) / (t_last[-1] - t_last[0]) if t_last[-1] > t_last[0] else 0.5
                for i, (xi, yi, ti) in enumerate(zip(x_last, y_last, t_norm)):
                    marker_opacity = 0.3 + 0.7 * ti  # Higher opacity for later points
                    ax.plot(xi, yi, '.', color=colors[name], markersize=2, 
                            alpha=marker_opacity, zorder=4)
            
            # Add a black bob (circle) at the end position
            ax.plot(x_smooth[-1], y_smooth[-1], 'o', color='black', markersize=8)
            
        else:
            # Not enough points for interpolation, plot line segments
            ax.plot(x_last, y_last, '-', 
                    color=colors[name], linewidth=3.0,
                    label=name)
            
            # If requested, mark all solution points with markers
            if mark_sol_points:
                ax.plot(x_last, y_last, '.', color=colors[name], 
                        markersize=2, alpha=0.7, zorder=4)
                
            # Add a black bob (circle) at the end position
            ax.plot(x_last[-1], y_last[-1], 'o', color='black', markersize=8)
        
        # Plot reference solution if available (more transparent and behind)
        if reference_sol is not None:
            ref_x, ref_y = spring_pendulum_positions(reference_sol.y[0], reference_sol.y[1])
            
            ref_window_indices = np.where((reference_sol.t >= start_time) & (reference_sol.t <= end_time))[0]
            
            if len(ref_window_indices) > 0:
                ref_x_last = ref_x[ref_window_indices]
                ref_y_last = ref_y[ref_window_indices]
                
                if len(ref_window_indices) > 3:
                    # Use cubic interpolation for reference too
                    ref_dx = np.diff(ref_x_last)
                    ref_dy = np.diff(ref_y_last)
                    ref_distances = np.sqrt(ref_dx**2 + ref_dy**2)
                    ref_cumulative_dist = np.concatenate(([0], np.cumsum(ref_distances)))
                    
                    ref_sample_pts = np.linspace(ref_cumulative_dist[0], ref_cumulative_dist[-1], 
                                               num=max(100, 5*len(ref_window_indices)))
                    
                    ref_f_x = interp1d(ref_cumulative_dist, ref_x_last, kind='cubic')
                    ref_f_y = interp1d(ref_cumulative_dist, ref_y_last, kind='cubic')
                    
                    ref_x_smooth = ref_f_x(ref_sample_pts)
                    ref_y_smooth = ref_f_y(ref_sample_pts)
                    
                    # Plot reference solution with transparent black - same thickness as main solution
                    ax.plot(ref_x_smooth, ref_y_smooth, '-', 
                            color='black', linewidth=3.0,
                            alpha=0.25, zorder=1)  # Lower zorder to put behind other solution
                    
                    # Add a transparent black bob at the end position
                    ax.plot(ref_x_smooth[-1], ref_y_smooth[-1], 'o', 
                            color='black', markersize=8, alpha=0.25, zorder=1)
                    
                    # Add a dashed reference pendulum bar (same thickness)
                    ax.plot([0, ref_x_smooth[-1]], [0, ref_y_smooth[-1]], '--', 
                            color='black', alpha=0.25, linewidth=2.5, zorder=1)
                else:
                    # Plot reference solution with transparent black - same thickness as main solution
                    ax.plot(ref_x_last, ref_y_last, '-', 
                            color='black', linewidth=3.0,
                            alpha=0.25, zorder=1)  # Lower zorder to put behind other solution
                    
                    # Add a transparent black bob at the end position
                    ax.plot(ref_x_last[-1], ref_y_last[-1], 'o', 
                            color='black', markersize=8, alpha=0.25, zorder=1)
                    
                    # Add a dashed reference pendulum bar (same thickness)
                    ax.plot([0, ref_x_last[-1]], [0, ref_y_last[-1]], '--', 
                            color='black', alpha=0.25, linewidth=2.5, zorder=1)
        
        # Plot the pendulum bar (thicker now)
        if len(x_last) > 0:
            # Plot the spring as a thicker solid black line from origin to last position
            ax.plot([0, x_last[-1]], [0, y_last[-1]], '-', color='black', alpha=0.9, linewidth=2.5, zorder=3)
        
        # Set up the position plot
        # ax.set_aspect('equal')
        # Set reasonable limits based on r values
        # r_max = np.max(sol.y[1]) * 1.1
        # ax.set_xlim(-r_max, r_max)
        # ax.set_ylim(-r_max, r_max)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_analysis_over_initial_conditions(solutions_list, compute_energy, initial_energies, start_time=0, end_time=None, time_threshold=None, timings=None, plot_std=False):
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
    start_time, end_time = _determine_time_window(solutions_list[0], start_time, end_time, time_threshold)
    
    # Colors for different methods - maintain consistent coloring across plots
    method_names = set()
    for solutions in solutions_list:
        method_names.update(solutions.keys())
    
    # Sort method names to ensure consistent color assignment
    method_names = sorted(list(method_names))
    
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(method_names)}
    colors['Reference'] = 'black'
    
    # Create figure with subplots
    n_plots = 3 if timings else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # If we have only one subplot, wrap it in a list
    if n_plots == 1:
        axes = [axes]
        
    ax_error = axes[0]   # Mean absolute error plot
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
            if 'reference' in name.lower() and sol.success:
                reference_sol = sol
                reference_name = name
                break
        
        if reference_sol is not None:
            reference_sols.append((reference_sol, reference_name))
            
            # For each method, compute error data
            for name, sol in solutions.items():
                if sol.success and name != reference_name:
                    if name not in methods_data:
                        methods_data[name] = {'error_data': [], 'energy_data': []}
                    
                    # Compute mean absolute error between this solution and reference
                    # Create interpolation functions for all state variables
                    ref_interp = []
                    for j in range(4):  # 4 state variables
                        ref_interp.append(interp1d(reference_sol.t, reference_sol.y[j], 
                                                kind='cubic', bounds_error=False, fill_value="extrapolate"))
                    
                    # Interpolate the reference solution to this solution's time points
                    error_sum = np.zeros_like(sol.t)
                    for j in range(4): 
                        interp_ref = ref_interp[j](sol.t)
                        error_sum += np.abs(sol.y[j] - interp_ref)
                    
                    # Calculate the mean absolute error across all state variables
                    mean_abs_error = error_sum / 4.0
                    
                    # Store the error data for this method and initial condition
                    methods_data[name]['error_data'].append((sol.t, mean_abs_error))
    
    # Calculate and store energy errors for each method and initial condition
    for i, (solutions, initial_energy) in enumerate(zip(solutions_list, initial_energies)):
        reference_name = reference_sols[i][1] if i < len(reference_sols) else None
        
        for name, sol in solutions.items():
            if sol.success and name != reference_name:  # Skip reference solution for energy error plot
                # Calculate energy at each time point
                y_data = sol.y.T
                energy = compute_energy(y_data)
                energy_error = np.abs(energy - initial_energy)
                
                # Store the energy error data
                if name in methods_data:
                    methods_data[name]['energy_data'].append((sol.t, energy_error))
    
    # Plot mean absolute error with shaded std dev regions
    for name, data in methods_data.items():
        if data['error_data']:
            # For each time series, interpolate to a common time grid
            common_t = np.linspace(start_time, end_time, 1000)
            interpolated_errors = []
            
            for t, error in data['error_data']:
                # Filter time points within our window
                mask = (t >= start_time) & (t <= end_time)
                if np.any(mask):
                    t_filtered = t[mask]
                    error_filtered = error[mask]
                    
                    # Create interpolation function and interpolate to common grid
                    if len(t_filtered) > 1:
                        error_interp = interp1d(t_filtered, error_filtered, kind='cubic', 
                                              bounds_error=False, fill_value="extrapolate")
                        interpolated_errors.append(error_interp(common_t))
            
            # Calculate mean and std dev across all initial conditions
            if interpolated_errors:
                interpolated_errors = np.array(interpolated_errors)
                mean_error = np.mean(interpolated_errors, axis=0)
                std_error = np.std(interpolated_errors, axis=0)
                
                # Plot mean line
                ax_error.semilogy(common_t, mean_error, '-', 
                              color=colors[name], label=name, alpha=0.8, linewidth=2)
                
                # Plot shaded region for std dev
                if plot_std:
                    ax_error.fill_between(common_t, mean_error - std_error, mean_error + std_error,
                                        color=colors[name], alpha=0.2)
    
    # Plot energy error with shaded std dev regions
    for name, data in methods_data.items():
        if data['energy_data']:
            # For each time series, interpolate to a common time grid
            common_t = np.linspace(start_time, end_time, 1000)
            interpolated_energy_errors = []
            
            for t, energy_error in data['energy_data']:
                # Filter time points within our window
                mask = (t >= start_time) & (t <= end_time)
                if np.any(mask):
                    t_filtered = t[mask]
                    error_filtered = energy_error[mask]
                    
                    # Create interpolation function and interpolate to common grid
                    if len(t_filtered) > 1:
                        error_interp = interp1d(t_filtered, error_filtered, kind='linear', 
                                              bounds_error=False, fill_value="extrapolate")
                        interpolated_energy_errors.append(error_interp(common_t))
            
            # Calculate mean and std dev across all initial conditions
            if interpolated_energy_errors:
                interpolated_energy_errors = np.array(interpolated_energy_errors)
                mean_error = np.mean(interpolated_energy_errors, axis=0)
                std_error = np.std(interpolated_energy_errors, axis=0)
                
                # Plot mean line
                ax_energy.semilogy(common_t, mean_error, '-', 
                               color=colors[name], label=name, alpha=0.8, linewidth=2)
                
                # # Plot shaded region for std dev
                # ax_energy.fill_between(common_t, mean_error - std_error, mean_error + std_error,
                #                      color=colors[name], alpha=0.2)
    
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
        methods = [name for name in methods_data.keys() if 'reference' not in name.lower()]
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
        ax_perf.bar(range(len(methods)), means, 
                yerr=stds,
                color=[colors.get(name) for name in methods],
                capsize=5)
        
        # Set up the performance plot
        ax_perf.set_title('Wall Clock Time')
        ax_perf.set_ylabel('Time (seconds)')
        ax_perf.set_xticks(range(len(methods)))
        ax_perf.set_xticklabels(methods, rotation=30, ha='right')
        ax_perf.grid(True, alpha=0.3)
    
    # Set up the error plot
    ax_error.set_title('Mean Absolute Error')
    ax_error.set_xlabel('Time')
    ax_error.grid(True, alpha=0.3)
    
    # Set up the energy error plot
    ax_energy.set_xlabel('Time')
    ax_energy.set_title('Energy error')
    ax_energy.grid(True, alpha=0.3)
    
    # Add legends - place all the way to the right outside plots
    if timings:
        # For 3-panel plot, place legend to the right of all plots
        fig.tight_layout(rect=[0, 0.05, 0.85, 1])  # Make space for the legend and rotated labels
        handles, labels = ax_energy.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.84, 0.65))
    else:
        # For 2-panel plot
        ax_energy.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    return fig

def create_spring_pendulum_animation(solutions, compute_energy, initial_energy, filename='spring_pendulum_animation.gif', fps=15, time_step=0.1):
    """
    Create an animated GIF of the spring pendulum simulation with multiple methods.
    
    Parameters:
    -----------
    solutions : dict
        Dictionary mapping method names to their solution objects from solve_ivp
    compute_energy : function
        Function to compute energy given a state
    initial_energy : float
        The initial energy of the system
    filename : str, optional
        The filename to save the GIF animation (default: 'spring_pendulum_animation.gif')
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
    plt.rcParams.update({
        'font.size': FONTSIZE,
        'axes.labelsize': FONTSIZE,
        'axes.titlesize': FONTSIZE,
        'xtick.labelsize': FONTSIZE,
        'ytick.labelsize': FONTSIZE,
        'legend.fontsize': FONTSIZE - 2,
        'figure.titlesize': FONTSIZE + 2,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.6,
    })
    
    # Colors for different methods
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {name: color_cycle[i % len(color_cycle)] for i, name in enumerate(solutions.keys())}
    colors['Reference'] = 'black'
    
    # Create a figure with 3 subplots
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 1.2])
    ax_pos = fig.add_subplot(gs[0])      # Position plot
    ax_time = fig.add_subplot(gs[1])     # Time series plot (r vs time)
    ax_energy = fig.add_subplot(gs[2])   # Energy error plot
    
    # Find the common time range for animation (use min of max times)
    max_time = min([sol.t[-1] for name, sol in solutions.items() if sol.success])
    
    # Create time points for animation
    anim_times = np.arange(0, max_time, time_step)
    
    # Find max radius for axis limits
    max_r = 0
    for name, sol in solutions.items():
        if sol.success:
            max_r = max(max_r, np.max(sol.y[1]))
    
    # Set up the position plot
    ax_pos.set_aspect('equal')
    ax_pos.set_xlim(-max_r * 1.1, max_r * 1.1)
    ax_pos.set_ylim(-max_r * 1.1, max_r * 1.1)
    ax_pos.set_title('Spring Pendulum Position')
    ax_pos.set_xlabel('x position')
    ax_pos.set_ylabel('y position')
    ax_pos.grid(True, alpha=0.3)
    
    # Set up the time plot
    ax_time.set_title('Radius vs Time')
    ax_time.set_xlabel('Time (seconds)')
    ax_time.set_ylabel('Radius')
    ax_time.set_xlim(0, max_time)
    ax_time.set_ylim(0, max_r * 1.1)
    ax_time.grid(True, alpha=0.3)
    
    # Set up the energy error plot
    ax_energy.set_xlabel('Time')
    ax_energy.set_ylabel('Energy Error (absolute)')
    ax_energy.set_title('Energy Conservation')
    ax_energy.set_xlim(0, max_time)
    ax_energy.set_yscale('log')
    
    # Find maximum energy error for setting y-limits
    max_error = 1e-1  # Default
    min_error = 1e-14 # Default
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
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14)
    
    # Initialize the pendulum artists for each solution
    pendulum_artists = {}
    spring_artists = {}
    radius_artists = {}
    energy_artists = {}
    history_artists = {}
    history_length = int(2 / time_step)  # Keep 2 seconds of history
    
    for name, sol in solutions.items():
        if sol.success:
            # Initialize pendulum with lines and markers
            spring, = ax_pos.plot([], [], '--', color='gray', alpha=0.7, lw=1)  # Spring
            point, = ax_pos.plot([], [], 'o', color=colors[name], markersize=10, alpha=0.7, label=name)  # Pendulum bob
            
            pendulum_artists[name] = point
            spring_artists[name] = spring
            
            # Time series line (radius vs time)
            radius_line, = ax_time.plot([], [], '-', color=colors[name], lw=2, label=name)
            radius_artists[name] = radius_line
            
            # Energy error line
            line_e, = ax_energy.plot([], [], '-', color=colors[name], lw=2, label=name)
            energy_artists[name] = line_e
            
            # Position history (trail)
            history, = ax_pos.plot([], [], '-', color=colors[name], alpha=0.3, lw=1)
            history_artists[name] = history
    
    # Add a fixed point at origin for pendulum pivot
    ax_pos.plot(0, 0, 'ko', markersize=6)
    
    # Add legends
    ax_pos.legend(loc='upper right')
    ax_time.legend(loc='upper right')
    ax_energy.legend(loc='upper right')
    
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
            original_energy = np.array([compute_energy(y_unique[:, i]) for i in range(y_unique.shape[1])])
            original_energy_error = np.abs(original_energy - initial_energy)
            
            if len(t_unique) > 1:  # Need at least 2 points for interpolation
                # Interpolate state variables
                y_interp = np.zeros((4, len(anim_times)))
                for i in range(4):  
                    interp_func = interp1d(t_unique, y_unique[i], kind='linear', bounds_error=False, fill_value="extrapolate")
                    y_interp[i] = interp_func(anim_times)
                
                # Compute positions for all animation time points
                x, y = spring_pendulum_positions(y_interp[0], y_interp[1])
                
                # Interpolate the pre-computed energy errors
                # This preserves the actual energy conservation properties of each method
                energy_error_interp = interp1d(t_unique, original_energy_error, kind='linear', 
                                             bounds_error=False, fill_value="extrapolate")(anim_times)
                
                # Store all interpolated data
                interp_data[name] = {
                    'r': y_interp[0], 'theta': y_interp[1],
                    'x': x, 'y': y, 'energy_error': energy_error_interp
                }
    
    # Define the animation update function
    def update(frame):
        t = anim_times[frame]
        time_text.set_text(f'Time: {t:.2f}s')
        
        artists = []
        
        for name in solutions.keys():
            if name in interp_data:
                data = interp_data[name]
                
                # Update pendulum position
                x, y = data['x'][frame], data['y'][frame]
                r = data['r'][frame]
                
                # Update pendulum point and spring
                pendulum_artists[name].set_data([x], [y])
                spring_artists[name].set_data([0, x], [0, y])
                
                # Update radius time series
                radius_line = radius_artists[name]
                visible_times = anim_times[:frame+1]
                visible_r = data['r'][:frame+1]
                radius_line.set_data(visible_times, visible_r)
                
                # Update energy error
                line_e = energy_artists[name]
                visible_errors = data['energy_error'][:frame+1]
                line_e.set_data(visible_times, visible_errors)
                
                # Update position history (trail)
                history = history_artists[name]
                start_idx = max(0, frame - history_length)
                history.set_data(data['x'][start_idx:frame+1], data['y'][start_idx:frame+1])
                
                artists.extend([pendulum_artists[name], spring_artists[name], 
                               radius_line, line_e, history])
        
        artists.append(time_text)
        return artists
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(anim_times), interval=1000/fps, blit=True)
    plt.tight_layout()
    
    # Save the animation if filename is provided
    if filename:
        try:
            anim.save(filename, writer='pillow', fps=fps, dpi=72)
            print(f"Animation saved to {filename}")
        except (IOError, ValueError, RuntimeError) as e:
            print(f"Failed to save animation: {e}")
    
    return anim



def find_small_r_events(solution, window_len_minus=5, window_len_plus=5, threshold=2, verbose=True):
    """
    Identify events where the radial distance r goes below a certain threshold.
    Returns a list of (start_time, middle_time, end_time, min_r) for each event.
    """
    import numpy as np

    r_thresh = np.min(solution.y[1])
    r_thresh *= threshold

    t_values = solution.t
    r_values = solution.y[1]

    # Find where r is below the threshold
    below_thresh_indices = np.where(r_values < r_thresh)[0]

    # Group consecutive indices into events
    events = []
    current_event = [below_thresh_indices[0]]

    for i in range(1, len(below_thresh_indices)):
        if below_thresh_indices[i] - below_thresh_indices[i-1] <= 2:  # Consider consecutive or nearly consecutive indices as same event
            current_event.append(below_thresh_indices[i])
        else:
            events.append(current_event)
            current_event = [below_thresh_indices[i]]

    # Add the last event
    if current_event:
        events.append(current_event)

    # Create 10-second windows centered around the middle of each event
    windows = []
    for event in events:
        # Get the middle index of the event
        middle_idx = event[len(event) // 2]
        middle_time = t_values[middle_idx]
        
        # Create a window centered at this time
        window_start = middle_time - window_len_minus
        window_end = middle_time + window_len_plus
        
        # Ensure window boundaries are within simulation time
        window_start = max(window_start, t_values[0])
        window_end = min(window_end, t_values[-1])
        
        # Find minimum r value during this event
        event_min_r = min(r_values[event])
        
        windows.append({
            'start_time': window_start,
            'middle_time': middle_time,
            'end_time': window_end,
            'min_r': event_min_r
        })

    if verbose:
        print("\n10-second windows centered around times when r < {:.3f}:".format(r_thresh))
        print("-------------------------------------------------------------")
        print(f"{'Window Start':>15} | {'Middle':>10} | {'Window End':>15} | {'Min r':>10}")
        print("-------------------------------------------------------------")

        for i, window in enumerate(windows):
            print(f"{window['start_time']:15.2f} | {window['middle_time']:10.2f} | {window['end_time']:15.2f} | {window['min_r']:10.5f}")

    return windows