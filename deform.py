import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def deform(xi, start, length, tau):
    xi1 = np.copy(np.asarray(xi))
    n_dims = xi1.shape[1]  
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(A.T @ A)
    U = np.zeros(length)
    gamma = np.zeros((length, n_dims))
    for idx in range(n_dims):
        U[0] = tau[idx]
        gamma[:,idx] = R @ U
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end,:] += gamma[0:end-start,:]
    return xi1

def visualize_deformation(original_traj, deformed_traj, tau):
    """
    Visualize original and deformed trajectories.
    
    Args:
        original_traj: Original trajectory (N x D) where D is number of dimensions
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


if __name__ == "__main__":
    n_timesteps = 25
    n_dims = 3  # Variable number of dimensions
    
    # Initialize trajectories dictionary
    trajectories = {}
    
    # 1. Circular motion (curve with smooth motion)
    t = np.linspace(0, 1.5*np.pi, n_timesteps)  # Changed to not complete full circle
    traj1 = np.zeros((n_timesteps, n_dims))
    traj1[:, 0] = np.cos(t)
    if n_dims > 1:
        traj1[:, 1] = np.sin(t)
    if n_dims > 2:
        traj1[:, 2] = 0.2 * t
    if n_dims > 3:
        traj1[:, 3:] = np.random.normal(0, 0.3, size=(n_timesteps, n_dims - 3))
    trajectories['Circular Motion'] = traj1
    
    # 2. Straight line (no curves)
    traj2 = np.zeros((n_timesteps, n_dims))
    traj2[:, 0] = np.linspace(0, 5, n_timesteps)  # Linear increase
    if n_dims > 1:
        traj2[:, 1] = np.linspace(0, 3, n_timesteps)
    if n_dims > 2:
        traj2[:, 2] = np.linspace(0, 2, n_timesteps)
    if n_dims > 3:
        traj2[:, 3:] = np.random.normal(0, 0.3, size=(n_timesteps, n_dims - 3))
    trajectories['Straight Line'] = traj2
    
    # 3. Closed loop (same start and end point)
    t = np.linspace(0, 3*np.pi, n_timesteps)  # Changed to not return to start
    traj3 = np.zeros((n_timesteps, n_dims))
    traj3[:, 0] = np.cos(t) * (1 + 0.3 * np.sin(3*t))
    if n_dims > 1:
        traj3[:, 1] = np.sin(t) * (1 + 0.3 * np.cos(3*t))
    if n_dims > 2:
        traj3[:, 2] = 0.1 * np.sin(4*t)
    if n_dims > 3:
        traj3[:, 3:] = np.random.normal(0, 0.3, size=(n_timesteps, n_dims - 3))
    trajectories['Curved Path'] = traj3
    
    # 4. Spiral (curves with increasing height)
    t = np.linspace(0, 3*np.pi, n_timesteps)
    traj4 = np.zeros((n_timesteps, n_dims))
    radius = np.linspace(1, 0.2, n_timesteps)
    traj4[:, 0] = radius * np.cos(t)
    if n_dims > 1:
        traj4[:, 1] = radius * np.sin(t)
    if n_dims > 2:
        traj4[:, 2] = np.linspace(0, 3, n_timesteps)
    if n_dims > 3:
        traj4[:, 3:] = np.random.normal(0, 0.3, size=(n_timesteps, n_dims - 3))
    trajectories['Spiral'] = traj4
    
    # 5. S-curve (wave-like motion)
    t = np.linspace(0, 3*np.pi, n_timesteps)  # Changed from 4*π to 3*π
    traj5 = np.zeros((n_timesteps, n_dims))
    traj5[:, 0] = np.linspace(0, 5, n_timesteps)
    if n_dims > 1:
        traj5[:, 1] = np.sin(t)
    if n_dims > 2:
        traj5[:, 2] = np.cos(2*t) * 0.5
    if n_dims > 3:
        traj5[:, 3:] = np.random.normal(0, 0.3, size=(n_timesteps, n_dims - 3))
    trajectories['S-Curve'] = traj5
    
    # 6. Zigzag (sharp changes, no smoothness)
    traj6 = np.zeros((n_timesteps, n_dims))
    angles = np.linspace(0, 4*np.pi, n_timesteps)
    traj6[:, 0] = np.linspace(0, 5, n_timesteps) + 0.5 * np.sign(np.sin(angles))
    if n_dims > 1:
        traj6[:, 1] = np.linspace(0, 3, n_timesteps) + 0.5 * np.sign(np.cos(angles))
    if n_dims > 2:
        traj6[:, 2] = np.linspace(1, -1, n_timesteps)
    if n_dims > 3:
        traj6[:, 3:] = np.random.normal(0, 0.3, size=(n_timesteps, n_dims - 3))
    trajectories['Zigzag'] = traj6
    
    # Apply deformation to each trajectory
    print("\n" + "="*60)
    print("TRAJECTORY DEFORMATION ANALYSIS")
    print("="*60)
    
    fig_main = plt.figure(figsize=(18, 12))
    
    # Use different deformation lengths for each trajectory
    deform_lengths = [25, 15, 20, 18, 22, 16]  # Different lengths for variety
    
    for idx, (name, traj) in enumerate(trajectories.items(), 1):
        # Create deformation parameters with variable dimensions
        tau = np.random.uniform(-0.4, 0.4, size=n_dims)
        deform_len = deform_lengths[idx - 1]
        traj_deformed = deform(traj.copy(), 0, deform_len, tau)
        
        # Statistics
        max_change = np.max(np.abs(traj - traj_deformed))
        mean_change = np.mean(np.abs(traj - traj_deformed))
        start_end_diff_orig = np.linalg.norm(traj[-1, :] - traj[0, :])
        start_end_diff_deformed = np.linalg.norm(traj_deformed[-1, :] - traj_deformed[0, :])
        
        print(f"\n{idx}. {name} (deform_length={deform_len}/{len(traj)})")
        print(f"   Max change: {max_change:.6f}")
        print(f"   Mean change: {mean_change:.6f}")
        print(f"   Original start-end distance (3D): {start_end_diff_orig:.6f}")
        print(f"   Deformed start-end distance (3D): {start_end_diff_deformed:.6f}")
        print(f"   τ range: [{np.min(tau):.4f}, {np.max(tau):.4f}]")
        
        # Subplot for visualization (adaptive based on dimensions)
        ax = fig_main.add_subplot(2, 3, idx)
        if n_dims >= 3:
            ax = fig_main.add_subplot(2, 3, idx, projection='3d')
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    'b-', label='Original', linewidth=2.5, alpha=0.8)
            ax.plot(traj_deformed[:, 0], traj_deformed[:, 1], traj_deformed[:, 2], 
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
            ax.plot(traj_deformed[:, 0], traj_deformed[:, 1], 
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
            ax.plot(range(len(traj_deformed)), traj_deformed[:, 0], 
                    'r--', label='Deformed', linewidth=2.5, alpha=0.8)
            ax.scatter(0, traj[0, 0], 
                       c='blue', s=150, marker='o', edgecolors='black', linewidth=2, zorder=5)
            ax.scatter(len(traj)-1, traj[-1, 0], 
                       c='green', s=150, marker='s', edgecolors='black', linewidth=2, zorder=5)
            ax.set_xlabel('Time step', fontsize=9)
            ax.set_ylabel('Value', fontsize=9)
        
        ax.set_title(f'{name}\n(deform_len={deform_len})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    print("\n" + "="*60)
    print("Displaying 3D trajectory comparisons...")
    print("="*60)
    plt.show()