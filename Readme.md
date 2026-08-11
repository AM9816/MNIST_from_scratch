# MNIST_no_ml_libs

A fully connected neural network written from scratch in C++, with a python frontend for easy and automatic data loading, and plug and play functionality. No PyTorch, TensorFlow or any other machine learning library, only Eigen for linear algebra. Compiles into a Windows DLL to be loaded with Python ctypes (see python example file).

## Current Features

- Fully implemented batched backpropagation with Kaiming parameter initialization, ReLU / leaky ReLU hidden layers and softmax + cross entropy output.
- SGD optimizer with momentum and decoupled weight decay, inverted dropout, minibatching and per-epoch hold-out validation.
- Large matrix operations parallelised through Eigen's OpenMP backend (requires /openmp)
- Parameter checkpointing using an efficient binary format, automatic saving, loading and format verification with a json header.
- Training runs primarily on a background thread, with python live polling current performance statistics for live benchmarking, shown as a tqdm progress bar.

## Requirements 

**C++**
- Windows, MSVC (Visual Studio 2022), x64
- C++ `20`
- [Eigen](https://eigen.tuxfamily.org/) 3.4.1

**Python**
- Python 3.13.0+
- numpy, tqdm, kagglehub
```bash
pip install numpy tqdm kagglehub
```

MNIST dataset is downloaded automatically via 'kagglehub' on initial run and cached locally.

## Compiling C++ into compatible DLL
1. Download Eigen source code such that `#include "Eigen/Dense"` resolves, either past Eigen folder into source code folder or add install directory to **C/C++ → General → Additional Include Directories**.
2. Set compilation configuration to **Release / x64**, Eigen heavily depends on optimizations, compiling with Debug increases running time by over an order of magnitude.
3. Enable **C/C++ → Language → Open MP Support (`/openmp`)** to parallelize large matrix operations.
4. Build, then pass the resulting DLL path to `C_FCN_Interface` in `c_init.py` in custom source file.



## Example Usage

```python
from c_init import C_FCN_Interface
from dataloader import load_or_download_MNIST

dataset = load_or_download_MNIST(N=1024, shuffle_=False)

C = C_FCN_Interface(r"path\to\MNIST_no_ml_libs.dll")
C.create_fcn([28*28, 512, 256, 128, 10], dropout=0.3, activ="relu")
C.add_dataset(dataset, bsize=128, classN=10)

C.train(epochs=10, lr=2e-4, momentum=0.85, weight_decay=2e-4)

print(C.test_accuracy())
C.save_params("latest.params")
```

`epochs=-1` trains indefinitely. Checkpoints are written at the end of each epoch.

## Misc Notes
- `using fpoint = float` in `datadefinitions.h` must match `c_float` in `c_init.py`, this is correctly configured by default but experimenting with different representation depths requires both to change.
