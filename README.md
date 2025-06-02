# MNIST Digit Recognition Neural Network

This project implements a neural network from scratch in C++ to recognize handwritten digits from the MNIST dataset.

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   ├── neural_network.h
│   └── mnist_loader.h
├── src/
│   ├── main.cpp
│   ├── neural_network.cpp
│   └── mnist_loader.cpp
└── data/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

## Requirements

- C++17 or later
- CMake 3.10 or later
- OpenCV (for image processing)
- MNIST dataset files

## Building the Project

1. Create a build directory:
```bash
mkdir build
cd build
```

2. Generate build files:
```bash
cmake ..
```

3. Build the project:
```bash
cmake --build .
```

## Running the Program

1. Make sure the MNIST dataset files are in the `data` directory:
   - train-images-idx3-ubyte
   - train-labels-idx1-ubyte
   - t10k-images-idx3-ubyte
   - t10k-labels-idx1-ubyte

2. Run the program:
```bash
./mnist_nn
```

## Features

- Neural network implementation with:
  - Input layer (784 neurons)
  - Hidden layer (30 neurons)
  - Output layer (10 neurons)
- Sigmoid activation function
- Softmax output layer
- Cross-entropy loss
- Mini-batch gradient descent
- Model saving and loading

## Performance

The neural network achieves approximately 95-97% accuracy on the MNIST test set after training for 10 epochs.

## License
This project was made by Adam Boufeid for his end of year project in ENSIT
