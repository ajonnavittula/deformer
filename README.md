# Trajectory Deformation

A Python implementation for smoothly deforming movement trajectories while maintaining smoothness constraints.

## Overview

This module implements a trajectory deformation technique based on:

> **Losey, D. P., et al.** "Trajectory Deformations from Physical Human-Robot Interaction." *IEEE Transactions on Robotics*, 2018.

The method creates smooth, realistic variations of motion trajectories by applying controlled perturbations while automatically maintaining smoothness through regularization. This is useful for:
- **Data augmentation**: Generate training data variants for robot learning
- **Trajectory sampling**: Create diverse trajectories around a nominal path
- **Motion adaptation**: Smoothly modify trajectories for different contexts

## How It Works

The `deform()` function takes a trajectory and applies controlled deformations:

### Algorithm
1. **Input**: A trajectory ξ (N timesteps × n dimensions) and deformation parameters τ
2. **Smoothness Matrix**: Constructs a discrete second-derivative constraint matrix A that penalizes non-smooth deformations
3. **Regularization**: Solves the least-squares problem to compute smooth deformation vectors:
   ```
   γ = (A^T A)^(-1) U
   ```
   where U is built from the input deformation parameters τ
4. **Output**: Applies the computed deformation γ to the trajectory

The key advantage is that the regularization ensures deformations remain smooth and realistic, even with large perturbations to the input parameters.


### Parameters
- **`xi` (array, shape N × D)**: Input trajectory with N timesteps and D dimensions

- **`tau` (array, shape D)**: Deformation magnitude per dimension. Typical range: **[-0.1, 0.1]**
  - Larger values create more noticeable deformations
  - Length must match number of dimensions in trajectory
  - Can be randomized for each trajectory variation
  
- **`length`**: Number of timesteps to deform
  - If `length < N`, only that portion is deformed (rest stays unchanged)
  - Allows partial trajectory modifications

- **`start`**: Starting timestep for deformation (often 0)

## Usage

### Basic Example

```python
import numpy as np
from deform import deform

# Create or load a trajectory (shape: N x D for any number of dimensions)
trajectory = np.random.randn(30, 7)  # 30 timesteps, 7 dimensions

# Define deformation parameters (one per dimension)
tau = np.random.uniform(-0.1, 0.1, size=7)

# Apply full trajectory deformation
deformed = deform(trajectory.copy(), start=0, length=30, tau=tau)

# Or deform only portion of trajectory
deformed_partial = deform(trajectory.copy(), start=0, length=15, tau=tau)

# Works with any dimensionality
trajectory_2d = np.random.randn(30, 2)  # 2D trajectories
tau_2d = np.random.uniform(-0.1, 0.1, size=2)
deformed_2d = deform(trajectory_2d.copy(), start=0, length=30, tau=tau_2d)

trajectory_10d = np.random.randn(30, 10)  # 10-dimensional trajectories
tau_10d = np.random.uniform(-0.1, 0.1, size=10)
deformed_10d = deform(trajectory_10d.copy(), start=0, length=30, tau=tau_10d)
```

### Visualization

Run the example to see 6 different trajectory types with deformations:

```bash
python deform.py
```

This displays a 2×3 grid showing:
- **Blue solid lines**: Original trajectories
- **Red dashed lines**: Deformed trajectories with different deformation lengths
- Markers: Start (blue circle) and end (green square) points

## Features

- **Mathematically smooth deformations**: Regularization enforces smoothness constraints
- **Partial trajectory modification**: Deform only a portion while keeping endpoints fixed
- **Dimension-wise control**: Independent deformation parameters for each dimension
- **Variable dimensionality**: Works with any number of dimensions (2D, 3D, 7-DOF robot arms, 10D+, etc.)
- **Data augmentation**: Easily generate trajectory variants for learning
- **Adaptive visualization**: Automatically visualizes 1D, 2D, or 3D trajectories based on dimensionality

## Requirements

- NumPy
- Matplotlib (for visualization)