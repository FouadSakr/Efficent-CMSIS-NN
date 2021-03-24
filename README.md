# Efficent CMSIS-NN

We propose an in-place computation strategy to reduce memory requirements of neural network inference. Our experimental analysis using CMSIS-NN library on the CIFAR-10 dataset shows that the  proposed memory optimization method can reduce the memory required by a NN model during execution by more than 27% and the total NN model occupance reduction is slightly more than 9%.

This optimization relies on https://github.com/ARM-software/ML-examples/tree/master/cmsisnn-cifar10 repository with slight modification on the code generation file.

## Modification on https://github.com/ARM-software/ML-examples/tree/master/cmsisnn-cifar10
We only updated "code_gen.py" to be compatible with our proposed optimization method. User will have first to download the above repository and only replace "code_gen.py" file with our optimized one (code_gen_optimized.py).
