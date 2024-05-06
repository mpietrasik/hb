# HB
Source code for our paper Hierarchical Blockmodelling for Knowledge Graphs. The implementation is an extension of Qirong Ho's Gibbs Sampler provided to the authors via private correspondence.

# Installation

1. Install Boost 1.67.0 C++ libraries
2. Install MinGW in MATLAB
3. Update the variables `BOOST_PATH` and `BOOST_LIB_PATH` in `compile.m` to reflect the root and library paths of Boost 1.67.0
4. Run `compile.m`

# Runtime Instructions

Run `run.m` to perform inference on the model on the Synthetic Binary Tree dataset. If you wish to perform inference using the other two datasets in our paper, follow these steps:

1. Change the size of the input matrix `E` in `compile.m`
2. Change the sizes of `model_N` and `model_R` in `gateway.cpp` to correspond to the size of the graph's adjacency tensor
3. Change the input file in the Mex gateway function `gateway.cpp`
4. Run `compile.m` to recompile C++ code
5. Update model hyperparameters
7. Run `run.m` to perform inference
