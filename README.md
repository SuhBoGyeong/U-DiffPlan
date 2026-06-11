<div align="center">

# U-DiffPlan

**Uncertainty-aware Diffusion-based Planning for Autonomous Racing**

**Accepted to IEEE Robotics and Automation Letters (RA-L) 2026**

</div>

---

## Overview

U-DiffPlan integrates a diffusion-based trajectory prediction model with Time-Optimal Model Predictive Control (TOMPC) for two-vehicle racing scenarios. The diffusion model predicts the opponent's future trajectory with uncertainty estimates, which the TOMPC incorporates to make robust overtaking and defensive decisions.

**Key Features:**

- **Trajectory Prediction**: UNet-based diffusion model predicts the opponent vehicle's future trajectory, outputting both mean predictions and uncertainty (log variance).
- **Uncertainty-aware TOMPC**: Acados/CasADi-based path-parametric TOMPC that incorporates prediction uncertainty into constraint tightening.
- **Game-theoretic Decision Making**: Both vehicles choose among four strategies via a utility-based decision framework:
  - **0: Following** - Track opponent with safe distance and drafting
  - **1: Overtaking** - Aggressive maneuver to pass opponent
  - **2: Driving** - Leader maintaining position on racing line
  - **3: Blocking** - Leader defending position against overtaking
- **Vehicle Dynamics**: Nonlinear bicycle model with Pacejka tire forces, dynamic road friction, and aerodynamic drafting effects.

---

## Project Structure

```
U-DiffPlan/
├── simulate.py                  # Run simulation with trained model
├── data_generation.py           # Generate training data from MPC racing
├── data_preprocessing_multi.py  # Preprocess raw simulation data
├── train.py                     # Train U-DiffPlan model
├── model.py                     # UDiffPlan model definition
├── train_func.py                # Training utilities (dataset, metrics, etc.)
├── train_generation_fun_modules_sr3.py  # UNet architecture (adapted from Palette)
├── plot_func.py                 # Plotting utilities
├── simulate_model.py            # Model loading for simulation
├── config_nn.py                 # Neural network configuration
├── LMS_Track_0.25.txt           # Track centerline data
├── environment.yml              # Conda environment specification
└── env/                         # Racing environment package
    ├── racing.py                # Gym environment (main simulation loop)
    ├── vehicle_model.py         # Vehicle dynamics model
    ├── decision_maker.py        # Game-theoretic decision making
    ├── config.py                # MPC & environment configuration
    ├── utils.py                 # Utility functions (RK4, Parameters)
    ├── renderer.py              # Frame rendering
    ├── live_plot.py             # Real-time visualization
    ├── mpc/                     # MPC controllers
    │   ├── mpc.py               # MPC with uncertainty support
    │   ├── mpc_baseline.py      # Baseline MPC (no uncertainty)
    │   ├── acados_ocp_pp.py     # Acados OCP formulation
    │   ├── acados_solver.py     # Acados solver wrapper
    │   ├── casadi_solver.py     # CasADi solver wrapper
    │   └── casadi_ocp_timeoptimal.py
    └── track/                   # Track representation
        └── track.py             # Spline-based track class
```

---

## Installation

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate U-DiffPlan
```

### 2. Install acados

acados must be installed separately:

```bash
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build && cd build
cmake .. -DACADOS_WITH_QPOASES=ON
make install -j4
cd ..
pip install -e interfaces/acados_template
```

For detailed instructions, see the [acados documentation](https://docs.acados.org/installation/).

---

## Usage

### Configuration

You can customize initial vehicle positions by editing `env/config.py`:

```python
# In Config class (for EV)
self.vehicle.Leader_thetaecephiV = [5.0, 0.0, 0.0, 1.0]      # [theta, ec, ephi, velocity]
self.vehicle.Follower_thetaecephiV = [3.0, 0.0, 0.0, 1.0]

# In Config_TV class (for TV)
self.vehicle.Leader_thetaecephiV = [5.0, 0.0, 0.0, 1.0]
self.vehicle.Follower_thetaecephiV = [3.0, 0.0, 0.0, 1.0]
```

**Parameters:**
- `theta`: Progress along track centerline [m]
- `ec`: Lateral deviation from centerline [m]
- `ephi`: Heading error [rad]
- `velocity`: Initial longitudinal velocity [m/s]

**Note:** These initial positions are used in `simulate.py`. For `data_generation.py`, initial positions are randomized for data diversity.

### Data Generation

Generate training data by running MPC-vs-MPC racing simulations:

```bash
# Generate data with EV starting behind (EV follower, TV leader)
python data_generation.py --num_samples 400 --save_dir data --scenario ev_behind

# Generate data with TV starting behind (TV follower, EV leader)
python data_generation.py --num_samples 400 --save_dir data --scenario tv_behind
```

**Scenario options:**
- `--scenario ev_behind`: EV starts behind → EV vx_max=1.5 (faster), TV vx_max=1.3 (slower)
- `--scenario tv_behind`: TV starts behind → TV vx_max=1.5 (faster), EV vx_max=1.3 (slower)
- No scenario specified: Use default config values

### Data Preprocessing

```bash
python data_preprocessing_multi.py
```

### Training

```bash
python train.py --gpu 0
```

### Simulation

Run simulation with the trained prediction model:

```bash
# EV as leader (EV vx_max=1.3, TV vx_max=1.5)
python simulate.py --gpu 0 --exp <experiment_name> --ev_leader y

# TV as leader (TV vx_max=1.3, EV vx_max=1.5)
python simulate.py --gpu 0 --exp <experiment_name> --ev_leader n
```

To run in data generation mode (MPC-only, no prediction model):

```bash
python simulate.py --data_gen_mode y --ev_leader y
```

**Leader/Follower configuration:**
- `--ev_leader y`: EV starts in front (leader gets lower speed limit for fair competition)
- `--ev_leader n`: TV starts in front

#### MPC Mode Selection

Use `--mpc_mode` to select the EV controller:

```bash
# Strategy MPC (default) — multi-mode MPCC with following/overtaking/driving/blocking
python simulate.py --mpc_mode strategy --ev_leader y

# Time-Optimal MPC — maximises track progress rate (dtheta) instead of tracking thetaref
python simulate.py --mpc_mode timeoptimal --ev_leader y
python simulate.py --mpc_mode timeoptimal --ev_leader n
```

---

## Acknowledgements

The UNet architecture in `train_generation_fun_modules_sr3.py` is adapted from [Palette: Image-to-Image Diffusion Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) (MIT License).
