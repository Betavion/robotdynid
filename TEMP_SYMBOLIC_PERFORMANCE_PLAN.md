# Symbolic Performance Optimization Plan

Date: 2026-05-20

Status: approved for execution

## 1. Objective

Improve the symbolic-performance path of `robotdynid` for:

1. symbolic inverse-dynamics construction
2. symbolic regressor construction
3. base-regressor projection
4. code generation latency
5. generated runtime efficiency for `C` and `C++`

The immediate bottleneck is the **construction path**, not just the final source-code rendering step.

## 2. Current Path

Current symbolic/codegen pipeline:

1. `symbolic.symbols.build_symbolic_context`
2. `symbolic.rne.build_inverse_dynamics`
3. `symbolic.regressor.build_standard_regressor`
4. `symbolic.base.build_base_regressor`
5. `codegen.cse_pipeline.apply_cse`
6. `codegen.c_codegen.generate_*`

This is correct but not optimal for large 6-DOF models.

## 3. Main Bottlenecks

### 3.1 CSE happens too late

The current code performs `sympy.cse` only after large symbolic matrices are already fully built.

Consequence:

- symbolic expression blow-up already happened before codegen starts
- `jacobian()` differentiates already-expanded expressions
- codegen CSE reduces emitted code size, but does not significantly reduce earlier construction cost

### 3.2 Construction path keeps large matrix expressions alive

`symbolic.rne.build_inverse_dynamics` currently propagates:

- joint transforms
- spatial transforms
- velocities
- accelerations
- forces

as large `SymPy` matrix expressions without a structured intermediate representation.

### 3.3 Blanket simplify in early stages

Current construction still applies costly simplification too early:

- `symbolic.spatial_math.axis_angle_rotation`
- `symbolic.rne._joint_spatial_transform`

These calls increase construction cost and do not guarantee the best graph shape for later reuse.

### 3.4 Helper blocks are too shallow semantically

Current `codegen` helper blocks are split only by count:

- `helper_block_size`

They are not split by:

- dependency layers
- physical phase
- expression hot spots

### 3.5 `H` and `tau` share the same export shape

Currently:

- `H` export and `tau` export both start from final symbolic matrices

But in practice:

- `H` is for identification/offline
- `tau` is for runtime/real-time use

They should be allowed to have different optimization strategies.

## 4. Literature / Reference Conclusions

### 4.1 SymPyBotics

Repository:

- https://github.com/cdsousa/SymPyBotics

Most relevant files:

- `sympybotics/symcode/subexprs.py`
- `sympybotics/robotmodel.py`
- `sympybotics/symcode/generation.py`

What is worth borrowing:

1. collecting subexpressions during generation, not only at the end
2. using an intermediate representation shaped like:
   - temporaries
   - outputs
3. reusing one subexpression pool across related generated artifacts

What should not be copied:

1. the DH/MDH-centered modeling assumptions
2. the regressor construction strategy based on one-hot substitution

### 4.2 SymPy Best Practices

The current plan should align with SymPy best practices:

1. avoid blanket `simplify` on large programmatic expressions
2. prefer targeted transforms
3. make structure explicit before final code printing

## 5. Target Architecture

Introduce an internal IR:

```text
SymbolicProgram
  -> blocks
    -> temporaries
    -> outputs
```

Suggested internal types:

```python
SymbolicTemporary
SymbolicBlock
SymbolicProgram
```

The IR should be:

1. internal-only at first
2. compatible with existing bundles during transition
3. consumable by both `C` and `C++` generators

## 6. Execution Plan

### Phase 1: Introduce IR

Deliverables:

- `symbolic/program.py`
- basic program/block/temporary dataclasses

Goals:

1. define a stable internal representation
2. keep existing public APIs working
3. allow codegen to consume either raw matrices or program blocks

### Phase 2: Remove early expensive simplify

Deliverables:

- update `symbolic.spatial_math`
- update `symbolic.rne`

Goals:

1. remove `simplify` from transform construction
2. replace with lighter operations only if needed
3. preserve existing numerical behavior

### Phase 3: Add early hoisting in RNE

Deliverables:

- structured forward/backward symbolic blocks

Targets to hoist:

- joint rotation
- homogeneous transform
- spatial transform
- `v_joint`
- `crm(v)`
- `crf(v)`
- `I * a`
- `I * v`

Goals:

1. shrink final `tau_total`
2. reduce `jacobian()` input complexity
3. create meaningful helper-block boundaries

### Phase 4: Rework regressor generation around IR

Deliverables:

- `tau_total` plus IR-backed program representation

Goals:

1. keep `jacobian()` as the main regressor path
2. structure the result as blocks/temporaries instead of a single flat matrix
3. avoid one-shot giant codegen input

### Phase 5: Make codegen consume IR natively

Deliverables:

- block-aware codegen path

Goals:

1. emit helper blocks by semantic phase
2. share temporaries between exported functions where possible
3. keep the current `C/C++` API surface stable

### Phase 6: Differentiate offline and runtime exports

Deliverables:

- separate optimization modes for `H` and `tau`

Modes:

1. generic export
2. specialized export with fixed `qds_star`

Goals:

1. keep `H` readable and identification-friendly
2. make `tau` smaller and faster for runtime use

## 7. Validation Strategy

After each phase:

1. run existing unit tests
2. compare symbolic numeric evaluation against Pinocchio
3. compare generated code numerical outputs against symbolic outputs

Required invariants:

1. `tau_symbolic == tau_pinocchio`
2. `H_symbolic == H_numeric_reference`
3. generated `C/C++` outputs match symbolic outputs

## 8. Risk Control

### Risk 1: break existing public APIs

Control:

- keep current bundle fields during the transition
- add new IR fields before removing old ones

### Risk 2: overengineering the IR

Control:

- first phase only introduces the minimum representation
- no early general-purpose DSL

### Risk 3: helper functions become too fragmented

Control:

- block granularity will remain configurable
- helper grouping will be driven by structure, not aesthetics

## 9. Immediate Next Step

Start with:

1. add internal symbolic IR types
2. remove early `simplify` from transform construction
3. keep tests green

That is the first concrete implementation step after this planning document.
