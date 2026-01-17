# shapelang: research compiler with executable GPU backends

**shapelang** is a proof-of-concept compiler toolchain exploring first-class ML semantics (tensor shapes, types, and autodiff) as language primitives rather than library conventions. It lowers a statically typed frontend representation to an existing production-grade GPU runtime (IREE) or a reference CPU interpreter.

This project exists to validate the feasibility of a shape-checked, GPU-native language frontend without reimplementing low-level kernel generation.

## Capabilities

*   **Backend Parity**: Ensures semantic equivalence between the reference CPU interpreter and the GPU backend (IREE/Vulkan).
*   **Compilation Pipeline**: Lowers typed IR to MLIR (linalg/arith dialects), which is then compiled to Vulkan SPIR-V via IREE.
*   **Autodiff**: Implements reverse-mode automatic differentiation as a source-to-source IR transformation with correctness verified against finite-difference checks.
*   **Artifact Management**: Provides deterministic compilation caching and explicit emission of intermediate representations (MLIR, VMFB) for inspection.
*   **Operational Discipline**: Enforces explicit failure modes and safe fallbacks rather than silent degradation.

## Non-Goals

*   **Not a Framework**: This is a language compiler, not a PyTorch/JAX competitor. It contains no dataset loaders or high-level model zoos beyond examples.
*   **Not a Benchmark**: Focus is on semantic correctness and compilation correctness, not peak FLOPs or kernel tuning.
*   **Not a Replacement**: It relies on IREE/MLIR for code generation and is not a replacement for CUDA or hand-written kernels.

## Golden Path: Regression

The `examples/regression/main.sl` program serves as the semantic anchor for the toolchain. It exercises:
1.  Tensor operations (`matmul`, `add`, `sub`, `mul`)
2.  Reverse-mode differentiation (`grad`)
3.  Gradient-based optimization loop
4.  Reduction operations (`mean`)

Successful execution of this program on both CPU and GPU backends with matching loss values validates the toolchain's correctness. Failure indicates a breakdown in semantic lowering or backend parity.

## Execution Model

```
Source (.sl) -> Parse -> Typed AST -> IR -> [Autodiff Pass] -> Backend Selection
                                                                    |
                                          +-------------------------+-------------------------+
                                          |                                                   |
                                     CPU Backend                                         GPU Backend
                                     (Interpreter)                                     (IREE / Vulkan)
                                          |                                                   |
                                     Direct Execution                                  Lower to MLIR
                                                                                              |
                                                                                       Compile to VMFB
                                                                                              |
                                                                                       IREE Runtime
```

## GPU Backend Notes

The GPU backend targets **Vulkan SPIR-V** via **IREE**. This choice allows for broad hardware compatibility (Linux/Windows, NVIDIA/AMD/Intel) without requiring a CUDA toolchain for the compiler itself.

*   **Requirements**: `iree-compile` and `iree-run-module` must be present in `PATH`.
*   **Target**: Currently hardcoded to `vulkan-spirv`.
*   **Platform**: Development and validation primarily target Linux environments; Windows support is experimental and toolchain-dependent.

## Usage

### Check Syntax and Types
```bash
sl check examples/regression/main.sl
```

### Compilation
```bash
# Compile to GPU artifact (VMFB)
sl compile examples/regression/main.sl --target gpu --out regression.vmfb

# Inspect intermediate MLIR
sl compile examples/regression/main.sl --target gpu --emit-backend-ir
```

### Execution
```bash
# Run on GPU (requires IREE tools)
sl run examples/regression/main.sl --device gpu

# Run on CPU (Reference)
sl run examples/regression/main.sl --device cpu
```

## Status

*   **Frontend**: Complete for supported subset (tensors, basic control flow, functions).
*   **CPU Backend**: Functional reference implementation.
*   **GPU Backend**: Functional via IREE. Linker issues may affect build on non-standard Windows environments; logic is complete.
*   **Autodiff**: Functional for basic ML operators (`matmul`, elementwise, `softmax`, `cross_entropy`). Control flow inside differentiating regions is limited.

## Audience

This repository is intended for compiler engineers, systems researchers, and language designers interested in the intersection of ML semantics and compiler infrastructure. It is not suitable for end-users looking to train models.
