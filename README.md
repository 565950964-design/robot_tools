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
