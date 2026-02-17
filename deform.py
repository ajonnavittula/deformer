import warnings
import numpy as np
import matplotlib.pyplot as plt


def deform(xi, start, length, tau):
    """
    Apply smooth trajectory deformation using regularization.
    
    Args:
        xi: Input trajectory (N x D array)
        start: Starting timestep for deformation
        length: Number of timesteps to deform
        tau: Deformation magnitude per dimension (D-element array)
    
    Returns:
        xi1: Deformed trajectory (N x D array)
    """
    xi1 = np.copy(xi)
    n_dims = xi1.shape[1]
    
    # Construct second-derivative smoothness constraint matrix
    A = np.zeros((length + 2, length))
    for i in range(length):
        A[i, i] = 1
        A[i + 1, i] = -2
        A[i + 2, i] = 1
    
    # Compute regularization matrix
    R = np.linalg.inv(A.T @ A)
    U = np.zeros(length)
    gamma = np.zeros((length, n_dims))
    
    # Compute smooth deformation vectors for each dimension
    for d in range(n_dims):
        U[0] = tau[d]
        gamma[:, d] = R @ U
    
    # Apply deformation to trajectory
    end = min(start + length, xi1.shape[0] - 1)
    xi1[start:end, :] += gamma[0:end - start, :]
    return xi1

def visualize_deformation(original_traj, deformed_traj, tau):
    """
    Visualize original and deformed trajectories.
    
    Args:
        original_traj: Original trajectory (N x D)
        deformed_traj: Deformed trajectory (N x D)
        tau: Deformation parameters used
    """
    n_dims = original_traj.shape[1]
    fig = plt.figure(figsize=(16, 10))
    
    # 3D trajectory comparison (if at least 3 dimensions available)
    if n_dims >= 3:
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        ax.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2], 
                'b-', label='Original', linewidth=2)
        ax.plot(deformed_traj[:, 0], deformed_traj[:, 1], deformed_traj[:, 2], 
                'r-', label='Deformed', linewidth=2)
        ax.scatter(original_traj[0, 0], original_traj[0, 1], original_traj[0, 2], 
                   c='blue', s=100, marker='o', label='Start')
        ax.scatter(original_traj[-1, 0], original_traj[-1, 1], original_traj[-1, 2], 
                   c='green', s=100, marker='s', label='End')
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
        ax.set_zlabel('Dim 2')
        ax.set_title('3D Trajectory Comparison')
        ax.legend()
    
    # 2D XY view (if at least 2 dimensions available)
    if n_dims >= 2:
        ax = fig.add_subplot(2, 2, 2)
        ax.plot(original_traj[:, 0], original_traj[:, 1], 'b-', label='Original', linewidth=2)
        ax.plot(deformed_traj[:, 0], deformed_traj[:, 1], 'r-', label='Deformed', linewidth=2)
        ax.scatter(original_traj[0, 0], original_traj[0, 1], c='blue', s=100, marker='o')
        ax.scatter(original_traj[-1, 0], original_traj[-1, 1], c='green', s=100, marker='s')
        ax.set_xlabel('Dim 0')
        ax.set_ylabel('Dim 1')
        ax.set_title('2D View (Dim 0-1 Plane)')
        ax.legend()
        ax.grid(True)
    
    # Individual dimension comparison
    ax = fig.add_subplot(2, 2, 3)
    timesteps = np.arange(len(original_traj))
    for i in range(n_dims):
        ax.plot(timesteps, original_traj[:, i], 'o-', alpha=0.5, label=f'Dim {i} (orig)', markersize=3)
        ax.plot(timesteps, deformed_traj[:, i], 's-', alpha=0.5, label=f'Dim {i} (deform)', markersize=3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.set_title(f'All {n_dims} Dimensions Over Time')
    ax.grid(True)
    
    # Deformation parameters
    ax = fig.add_subplot(2, 2, 4)
    ax.bar(np.arange(n_dims), tau)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Deformation Parameter (τ)')
    ax.set_title('Deformation Parameters Used')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _fill_higher_dims(trajectory, n_dims, scale=0.3):
    """Fill dimensions beyond the first 3 with random noise."""
    if n_dims > 3:
        trajectory[:, 3:] = np.random.normal(0, scale, size=(len(trajectory), n_dims - 3))


def _plot_trajectory_comparison(ax, traj, traj_def, n_dims, name, deform_len):
    """Plot trajectory comparison with adaptive dimensionality."""
    if n_dims >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(ax.get_position(), projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                'b-', label='Original', linewidth=2.5, alpha=0.8)
        ax.plot(traj_def[:, 0], traj_def[:, 1], traj_def[:, 2], 
                'r--', label='Deformed', linewidth=2.5, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], 
                   c='blue', s=150, marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], 
                   c='green', s=150, marker='s', edgecolors='black', linewidth=2, zorder=5)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
        ax.set_zlabel('Z', fontsize=9)
        ax.view_init(elev=20, azim=45)
    elif n_dims >= 2:
        ax.plot(traj[:, 0], traj[:, 1], 
                'b-', label='Original', linewidth=2.5, alpha=0.8)
        ax.plot(traj_def[:, 0], traj_def[:, 1], 
                'r--', label='Deformed', linewidth=2.5, alpha=0.8)
        ax.scatter(traj[0, 0], traj[0, 1], 
                   c='blue', s=150, marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], 
                   c='green', s=150, marker='s', edgecolors='black', linewidth=2, zorder=5)
        ax.set_xlabel('X', fontsize=9)
        ax.set_ylabel('Y', fontsize=9)
    else:
        ax.plot(range(len(traj)), traj[:, 0], 
                'b-', label='Original', linewidth=2.5, alpha=0.8)
        ax.plot(range(len(traj_def)), traj_def[:, 0], 
                'r--', label='Deformed', linewidth=2.5, alpha=0.8)
        ax.scatter(0, traj[0, 0], 
                   c='blue', s=150, marker='o', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(len(traj) - 1, traj[-1, 0], 
                   c='green', s=150, marker='s', edgecolors='black', linewidth=2, zorder=5)
        ax.set_xlabel('Time step', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
    
    ax.set_title(f'{name}\n(deform_len={deform_len})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)


if __name__ == "__main__":
    n_timesteps = 25
    n_dims = 3  # Variable number of dimensions (change to 2, 3, 10, etc. to test)
    
    # Define trajectory generators
    t = np.linspace(0, 1.5 * np.pi, n_timesteps)
    t_long = np.linspace(0, 3 * np.pi, n_timesteps)
    
    trajectories = {
        'Circular Motion': lambda: np.column_stack([
            np.cos(t),
            np.sin(t) if n_dims > 1 else np.zeros(n_timesteps),
            0.2 * t if n_dims > 2 else np.zeros(n_timesteps)
        ]),
        'Straight Line': lambda: np.column_stack([
            np.linspace(0, 5, n_timesteps),
            np.linspace(0, 3, n_timesteps) if n_dims > 1 else np.zeros(n_timesteps),
            np.linspace(0, 2, n_timesteps) if n_dims > 2 else np.zeros(n_timesteps)
        ]),
        'Curved Path': lambda: np.column_stack([
            np.cos(t_long) * (1 + 0.3 * np.sin(3 * t_long)),
            np.sin(t_long) * (1 + 0.3 * np.cos(3 * t_long)) if n_dims > 1 else np.zeros(n_timesteps),
            0.1 * np.sin(4 * t_long) if n_dims > 2 else np.zeros(n_timesteps)
        ]),
        'Spiral': lambda: np.column_stack([
            np.linspace(1, 0.2, n_timesteps) * np.cos(t_long),
            np.linspace(1, 0.2, n_timesteps) * np.sin(t_long) if n_dims > 1 else np.zeros(n_timesteps),
            np.linspace(0, 3, n_timesteps) if n_dims > 2 else np.zeros(n_timesteps)
        ]),
        'S-Curve': lambda: np.column_stack([
            np.linspace(0, 5, n_timesteps),
            np.sin(t_long) if n_dims > 1 else np.zeros(n_timesteps),
            np.cos(2 * t_long) * 0.5 if n_dims > 2 else np.zeros(n_timesteps)
        ]),
        'Zigzag': lambda: np.column_stack([
            np.linspace(0, 5, n_timesteps) + 0.5 * np.sign(np.sin(np.linspace(0, 4 * np.pi, n_timesteps))),
            np.linspace(0, 3, n_timesteps) + 0.5 * np.sign(np.cos(np.linspace(0, 4 * np.pi, n_timesteps))) if n_dims > 1 else np.zeros(n_timesteps),
            np.linspace(1, -1, n_timesteps) if n_dims > 2 else np.zeros(n_timesteps)
        ]),
    }
    
    # Create trajectories with proper dimensions
    traj_dict = {}
    for name, traj_fn in trajectories.items():
        traj = np.zeros((n_timesteps, n_dims))
        temp = traj_fn()
        traj[:, :min(3, n_dims)] = temp[:, :min(3, n_dims)]
        _fill_higher_dims(traj, n_dims)
        traj_dict[name] = traj
    
    print("\n" + "=" * 60)
    print("TRAJECTORY DEFORMATION ANALYSIS")
    print("=" * 60)
    
    # Create combined figure with all experiments
    fig_combined = plt.figure(figsize=(20, 14))
    
    # Top row: Varying start positions (first 4 subplots)
    print("\nVarying start position experiment:")
    demo_traj = traj_dict['Spiral']
    demo_tau = np.random.uniform(-0.4, 0.4, size=n_dims)
    deform_length = 12
    start_positions = [0, 3, 6, 9]
    
    for pos_idx, start_pos in enumerate(start_positions, 1):
        traj_var = deform(demo_traj.copy(), start_pos, deform_length, demo_tau)
        
        ax = fig_combined.add_subplot(3, 4, pos_idx)
        _plot_trajectory_comparison(ax, demo_traj, traj_var, n_dims, 
                                    f'Start={start_pos}', deform_length)
        
        max_change = np.max(np.abs(demo_traj - traj_var))
        print(f"  Start={start_pos}: Max change={max_change:.6f}")
    
    # Remaining rows: Main trajectory types (subplots 5-10, then 9-14 would overflow)
    # Use subplots 5-10 for the 6 main trajectories
    deform_lengths = [25, 15, 20, 18, 22, 16]
    
    for idx, (name, traj) in enumerate(traj_dict.items(), 1):
        tau = np.random.uniform(-0.4, 0.4, size=n_dims)
        deform_len = deform_lengths[idx - 1]
        traj_deformed = deform(traj.copy(), 0, deform_len, tau)
        
        # Compute statistics
        max_change = np.max(np.abs(traj - traj_deformed))
        mean_change = np.mean(np.abs(traj - traj_deformed))
        start_end_orig = np.linalg.norm(traj[-1, :] - traj[0, :])
        start_end_deformed = np.linalg.norm(traj_deformed[-1, :] - traj_deformed[0, :])
        
        print(f"\n{idx}. {name} (deform_length={deform_len}/{len(traj)})")
        print(f"   Max change: {max_change:.6f}")
        print(f"   Mean change: {mean_change:.6f}")
        print(f"   Start-end distance (original): {start_end_orig:.6f}")
        print(f"   Start-end distance (deformed): {start_end_deformed:.6f}")
        print(f"   τ range: [{np.min(tau):.4f}, {np.max(tau):.4f}]")
        
        # Place in subplots: first 4 are start positions, next 6 are trajectories
        subplot_idx = 4 + idx
        ax = fig_combined.add_subplot(3, 4, subplot_idx)
        _plot_trajectory_comparison(ax, traj, traj_deformed, n_dims, name, deform_len)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()
    
    print("\n" + "=" * 60)
    print("Displaying combined trajectory deformation analysis...")
    print("=" * 60)
    plt.show()