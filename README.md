# robotdynid

`robotdynid` is a Python library for identifying rigid-body dynamic parameters of serial robots from a `URDF` model and motion/torque data.

## Features

- Load serial robot models from `URDF`
- Build symbolic rigid-body inverse dynamics and regressors
- Compute numerical regressors with `Pinocchio`
- Select base inertial parameters numerically
- Identify inertial and friction parameters from:
  - a single CSV containing `q/qd/qdd/tau`
  - split motion and torque CSV files
- Export C code for base regressors and torque prediction

## Scope

The library focuses on **serial open-chain robots**.

## Installation

Install from a local checkout:

```bash
pip install .
```

Install in editable mode for development:

```bash
pip install -e .
```

Install directly from GitHub:

```bash
pip install git+https://github.com/Betavion/robotdynid.git
```

## Main API

Typical entrypoints:

- `robotdynid.io.load_robot_from_urdf`
- `robotdynid.numeric.build_pinocchio_model`
- `robotdynid.numeric.build_pinocchio_regressor_evaluator`
- `robotdynid.workflow.run_identification_workflow`
- `robotdynid.codegen.generate_base_regressor_c_function`

## Development

Run the test suite with:

```bash
python -m unittest
```

## License

MIT
