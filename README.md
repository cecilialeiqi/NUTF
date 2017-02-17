# Negative-Unlabeled Tensor Factorization for Location\\ Context Inference from Inaccurate Mobility Data (NUTF)

This repository includes a simple matlab simulation and C++ codes for NUTF

## Usage
* matlab version: run simulateDemo.m in ./matlab/

* cpp version: 

Install Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download) and put it in ./eigen/

Compile training file:
----------
make train

```
./train test.txt 14 6 3 5 5
```

Compile simulation file:
----------
make simulate
```
./simulate N T C r iter k
(N: number of users, T: number of time steps, C: number of locations, r: rank, iter: number of iterations, k: sparsity)
e.g.:
./simulate 10000 500 200 20 10 20
```
