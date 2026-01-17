# Autodiff Implementation

shapelang implements reverse-mode automatic differentiation as a source-to-source transformation on the typed IR.

## Supported Operations
- **Linear Algebra**: `matmul`
- **Elementwise**: `add`, `sub`, `mul`, `div`, `exp`, `log`, `relu`, `tanh`
- **Reductions**: `sum`, `mean`
- **Neural Network**: `softmax`, `cross_entropy_logits`

## Gradient Semantics
The compiler generates an adjoint function (suffixed with `_grad`) that returns a tuple containing the loss and gradients for all input parameters. Gradient accumulation handles variable reuse (fan-out) in the dataflow graph.
