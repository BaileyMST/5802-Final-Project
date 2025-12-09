This is a final project for CS5802 Intro to Parallel Programming and Algorithms at Missouri S&T
The code uses numpy and scipy to implement image convolution with the goal of blob detection, without the usage of OpenCV

convolve.py is the sequential version of the code. It is run with ```python convolve.py``` WARNING: This will take hours to run

mpi_convolve.py uses mpi4py to implement cpu-based parallelization. It is run with 
```mpirun -n {threads to run with} python mpi_convolve.py```

cuda_convolve.py uses CUDA to implement GPU parallelization. It is run with ```insert run command here```