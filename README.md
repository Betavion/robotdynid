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

## Install

```bash
python -m venv --without-pip .venv
./scripts/bootstrap_venv.sh
```

Or install directly:

```bash
pip install -e .
```

## Main API

Typical entrypoints:

- `robotdynid.io.load_robot_from_urdf`
- `robotdynid.numeric.build_pinocchio_model`
- `robotdynid.numeric.build_pinocchio_regressor_evaluator`
- `robotdynid.workflow.run_identification_workflow`
- `robotdynid.codegen.generate_base_regressor_c_function`

## License

MIT
