# MNIST_no_ml_libs

A fully connected neural network written from scratch in C++, with a python frontend for easy and automatic data loading, and plug and play functionality. No PyTorch, TensorFlow or any other machine learning library, only Eigen for linear algebra. Compiles into a Windows DLL to be loaded with Python ctypes (see python example file).

## Current Features

- Fully implemented batched backpropagation, matrices kept as large as possible to make use of Eigen's multithreaded matrix operations.
- SGD optimizer with momentum and weight decay, with minibatching and hold-out verification.
- Parameter checkpointing using an efficient binary format, automatic saving and loading.
- Training runs primarily on a background thread, with python live polling current performance statistics for live benchmarking.

## Requirements 

**C++**
- [Eigen](https://eigen.tuxfamily.org/) 3.4+

**Python**
- Python 3.8+
- numpy, tqdm, kagglehub
```bash
pip install numpy pillow tqdm kagglehub
```

MNIST dataset is downloaded automatically via 'kagglehub' on initial run and cached locally.

## Example Usage

```python
from c_init import C_FCN_Interface
from dataloader import load_or_download_MNIST

dataset = load_or_download_MNIST(N=1024, shuffle_=False)

C = C_FCN_Interface(r"path\to\MNIST_no_ml_libs.dll")
C.create_fcn([28*28, 512, 256, 128, 10], dropout=0.3, activ=1)
C.add_dataset(dataset, bsize=128, classN=10)

C.train(epochs=-1, lr=2e-4, momentum=0.85, weight_decay=2e-4)

print(C.test_accuracy())
C.save_params("latest.pt")
```

`epochs=-1` trains indefinitely. Checkpoints are written at the end of each epoch.


