This is a final project for CS5802 Intro to Parallel Programming and Algorithms at Missouri S&T
The code uses numpy and scipy to implement image convolution with the goal of blob detection, without the usage of OpenCV

convolve.py is the sequential version of the code

mpi_convolve.py uses mpi4py to implement cpu-based parallelization

cuda_convolve.py uses CUDA to implement GPU parallelization