```markdown
# Pinocchio IK Solver

## Installation & Setup

### Requirements
- Python 3.7+
- [Pinocchio](https://github.com/stack-of-tasks/pinocchio) >= 2.6.0
- NumPy, SciPy

```bash
pip install pinocchio numpy scipy
```

### Configuration

Edit these variables in `IK.py` before use:

```python
URDF_PATH = "/path/to/your/robot.urdf"

JOINT_NAMES = [
    "joint_1",  # Replace with your actual joint names from URDF
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7"
]

EE_LINK_NAME = "end_effector_link"  # Your end-effector link name
```

## Mathematical Methodology

### Problem Formulation

Given desired pose $\mathbf{T}_{des} \in SE(3)$, find joint angles $\mathbf{q}$ that minimize:

$$\mathbf{e} = \text{log}_6(\mathbf{T}_{current}^{-1} \mathbf{T}_{des}) \in \mathbb{R}^6$$

where $\mathbf{e} = [\mathbf{e}_p; \mathbf{e}_r]$ combines position and rotation error.

### Damped Least Squares (DLS)

At each iteration, solve for joint update $\Delta \mathbf{q}$:

$$\Delta \mathbf{q} = \mathbf{J}^\dagger \mathbf{e}$$

with damped pseudoinverse:

$$\mathbf{J}^\dagger = \mathbf{J}^T (\mathbf{J} \mathbf{W} \mathbf{J}^T + \lambda^2 \mathbf{I}_6)^{-1} \mathbf{W}$$

where:
- $\mathbf{J} \in \mathbb{R}^{6 \times n}$: Geometric Jacobian
- $\mathbf{W} = \text{diag}(w_p, w_p, w_p, w_r, w_r, w_r)$: Weight matrix for position/rotation priority
- $\lambda$: Damping factor

Update rule:
$$\mathbf{q}_{k+1} = \mathbf{q}_k + \alpha \cdot \Delta \mathbf{q}$$

with step size $\alpha \in (0,1]$ and joint limits clamping.

### Adaptive Damping

Damping adjusts dynamically based on residual error $\|\mathbf{e}\|$:

$$\lambda_{eff} = \lambda_{base} \cdot (1 + \gamma \cdot \min(\|\mathbf{e}\|, 1.0))$$

- High residual $\rightarrow$ increased damping (robustness in singularities)
- Low residual $\rightarrow$ reduced damping (convergence accuracy)

### Null-Space Optimization

For redundant manipulators ($n > 6$), project posture optimization into null space:

$$\Delta \mathbf{q}_{null} = (\mathbf{I}_n - \mathbf{J}^\dagger \mathbf{J}) \cdot \mathbf{K}_{null} (\mathbf{q}_{ref} - \mathbf{q})$$

Total update:
$$\Delta \mathbf{q}_{total} = \Delta \mathbf{q}_{task} + \Delta \mathbf{q}_{null}$$

This minimizes $\|\mathbf{q} - \mathbf{q}_{ref}\|$ without affecting end-effector pose.

### Reference Frames

**LOCAL Mode:**
- Error: $\mathbf{e}_{local} = \text{log}_6(\mathbf{T}_{current}^{-1} \mathbf{T}_{des})$
- Jacobian: $\mathbf{J}_{local}$ (body twist representation)
- Suitable for local adjustments

**WORLD Mode:**
- Error: $\mathbf{e}_{world} = \text{Ad}_{\mathbf{T}_{current}} \cdot \mathbf{e}_{local}$
- Jacobian: $\mathbf{J}_{world} = \text{Ad}_{\mathbf{T}_{current}} \cdot \mathbf{J}_{local}$
- Suitable for global trajectory tracking
```
