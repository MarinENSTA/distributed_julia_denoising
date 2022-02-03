# Smoother Julia Denoising

## Authors

The base code was written by Feriel ABBOUD, all further modifications, updates and improvement by Marin STAMM

## About this version of the project

**This is not the final version of the project**. If you need to use the final version, please refer to appropriate folder. Current folder aims to show the algorithm during its developpment steps.

The goal of this version was to handle better any number of initialised parallel processes. It was made in order to obtain smoother time-speedup curves and a better behaving algorithm evening workload between all processes.

As this proved to be succesful and useful, this is implemented in the final version of the algorithm.

## Julia installation 
This is implemented in julia 1.4.1

Please install MPI and HDF5 on your machine before installing the following Julia packages 

To install necessary packages, open a julia terminal and type :
```julia
using Pkg         

Pkg.add("Images")
Pkg.add("HDF5")    
Pkg.add("MAT")    
Pkg.add("MPI")
Pkg.add("Suppressor")
```

To run this projet on your machine with four parallel processes using mpi: 
```bash
mpirun -np 4 julia programMain.jl
```


