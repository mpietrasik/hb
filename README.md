# HB

Source code for our paper Hierarchical Blockmodelling for Knowledge Graphs. The implementation is an extension of Qirong Ho's Gibbs Sampler provided to the authors via private correspondence.

# Contents

* `compile.m` defines the necessary paths and the mex compilation and linking function
* `run.m` is the script to run the Gibbs sampler on the Synthetic Binary Tree dataset described in the paper
* `sampler.m` is the Matlab wrapper for the C++ Gibbs sampler. This wrapper calls the mex gateway function `gateway.cpp`
* `gateway.ccp` is the mex gateway function. It is called by `sampler.m` and initializes the Gibbs sampler
* `hb.hpp` is the core of the Gibbs sampler. It defines the tree structure and performs all the necessary sampling computations
* `data/synthetic_binary_tree.txt` is the for data file the Synthetic Binary Tree dataset. Each line contains the index of the subject, object, and predicate of a triple in the data
* `data/fb15k-237.txt` is the for data file the FB15k-237 dataset. Each line contains the index of the subject, object, and predicate of a triple in the data
* `data/wikidata.txt` is the for data file the Wikidata dataset. Each line contains the index of the subject, object, and predicate of a triple in the data
* `data/fb15k-237_id2entity.txt` is the transaltion file between indices and entities for the FB15k-237 dataset
* `data/fb15k-wikidata.txt` is the transaltion file between indices and entities for the Wikidata dataset

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
