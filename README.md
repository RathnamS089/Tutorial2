

# PyTorch Perceptron with Manual Initialization

This repository contains a reference implementation of a Multi-Layer Perceptron (MLP) using PyTorch.

Unlike standard implementations that utilize random weight initialization, this project explicitly hardcodes the weight matrices and bias vectors. This design is intended for educational purposes, specifically to allow students and developers to verify hand-calculated backpropagation steps against PyTorch's automatic differentiation engine.

## Project Overview

The model consists of a simple feed-forward neural network with the following architectural specifications:

* **Input Layer:** 2 Features
* **Hidden Layer:** 2 Neurons (Sigmoid Activation)
* **Output Layer:** 2 Neurons (Sigmoid Activation)
* **Optimization:** Stochastic Gradient Descent (SGD)
* **Loss Function:** Mean Squared Error (MSE)

## Dependencies

The only requirement for this project is the PyTorch library.

```bash
pip install torch

```

## Model Architecture & Parameters

To ensure deterministic behavior for debugging, the parameters are initialized to the specific values listed below.

### Layer 1 (Input to Hidden)

* **Weights:** `[[0.15, 0.20], [0.25, 0.30]]`
* **Bias:** `[0.35, 0.35]`

### Layer 2 (Hidden to Output)

* **Weights:** `[[0.40, 0.45], [0.50, 0.55]]`
* **Bias:** `[0.60, 0.60]`


## Implementation Details

### The `Perceptron` Class

The class inherits from `nn.Module`, which is the base class for all neural network modules in PyTorch.

* **`__init__`**: The constructor defines the layers. We access the `.data` attribute of the weights and biases to overwrite the default random initialization. This is critical for reproducing manual calculations.
* **`forward`**: This method defines the data flow. It includes optional print statements to display intermediate values (`net_h`, `out_h`, etc.), which assists in granular debugging.

### Common Issues

When modifying this code, ensure the following:

1. **Method Naming:** The function handling the data pass must be named `forward`. Using other names (e.g., `forward_sel`) will cause a `NotImplementedError` when calling `model(x)`.
2. **Return Statement:** The `forward` method must return the final output tensor. Omitting the return statement will result in `NoneType` errors during loss calculation.
3. **Tensor Shapes:** Ensure that manual weight tensors match the dimensions defined by the `nn.Linear` arguments.

## License

This project is open-source and available for educational use.
