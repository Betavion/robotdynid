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
- Export `C` and `C++` code for base regressors and torque prediction
- Generate ROS2-friendly `C++` kernel classes with:
  - no dynamic allocation in the hot path
  - explicit `double*` / `const double*` interfaces
  - optional helper blocks for temporary subexpressions

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
- `robotdynid.codegen.generate_base_regressor_cpp_function`
- `robotdynid.codegen.generate_prediction_cpp_function`

## Development

Run the test suite with:

```bash
python -m unittest
```

## Code generation

`robotdynid` supports two code-generation targets:

- `C`
  - plain functions for embedded or middleware-agnostic use
- `C++`
  - ROS2-friendly static kernel classes suitable for inclusion in `rclcpp` nodes or controllers

The `C++` backend avoids dynamic allocation in the generated runtime path and can split common subexpressions into helper functions when the generated expression graph is large enough.

## License

MIT
